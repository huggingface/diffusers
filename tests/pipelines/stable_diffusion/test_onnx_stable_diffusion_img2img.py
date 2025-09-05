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

import random
import unittest

import numpy as np

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    OnnxStableDiffusionImg2ImgPipeline,
    PNDMScheduler,
)

from ...testing_utils import (
    floats_tensor,
    is_onnx_available,
    load_image,
    nightly,
    require_onnxruntime,
    require_torch_gpu,
)
from ..test_pipelines_onnx_common import OnnxPipelineTesterMixin


if is_onnx_available():
    import onnxruntime as ort


class OnnxStableDiffusionImg2ImgPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    hub_checkpoint = "hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline"

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 128, 128), rng=random.Random(seed))
        generator = np.random.RandomState(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_pipeline_default_ddim(self):
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        assert np.abs(image_slice - expected_slice).max() < 1e-1

    def test_pipeline_pndm(self):
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config, skip_prk_steps=True)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.61737, 0.54642, 0.53183, 0.54465, 0.52742, 0.60525, 0.49969, 0.40655, 0.48154])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_lms(self):
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        # warmup pass to apply optimizations
        _ = pipe(**self.get_dummy_inputs())

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.52761, 0.59977, 0.49033, 0.49619, 0.54282, 0.50311, 0.47600, 0.40918, 0.45203])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_euler(self):
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.52911, 0.60004, 0.49229, 0.49805, 0.54502, 0.50680, 0.47777, 0.41028, 0.45304])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_euler_ancestral(self):
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.52911, 0.60004, 0.49229, 0.49805, 0.54502, 0.50680, 0.47777, 0.41028, 0.45304])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_dpm_multistep(self):
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.65331, 0.58277, 0.48204, 0.56059, 0.53665, 0.56235, 0.50969, 0.40009, 0.46552])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1


@nightly
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
            safety_checker=None,
            feature_extractor=None,
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
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler", revision="onnx"
        )
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            revision="onnx",
            scheduler=lms_scheduler,
            safety_checker=None,
            feature_extractor=None,
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
            num_inference_steps=20,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.8043, 0.926, 0.9581, 0.8119, 0.8954, 0.913, 0.7209, 0.7463, 0.7431])
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues

        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2
