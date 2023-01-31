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

import random
import unittest

import numpy as np

from diffusers.pipelines.stable_diffusion import OnnxStableDiffusionUpscalePipeline
from diffusers.utils import floats_tensor
from diffusers.utils.testing_utils import (
    is_onnx_available,
    load_image,
    nightly,
    require_onnxruntime,
    require_torch_gpu,
)

from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin


if is_onnx_available():
    import onnxruntime as ort


class OnnxStableDiffusionUpscalePipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    # TODO: is there an appropriate internal test set?
    hub_checkpoint = "/home/ssube/code/github/ssube/onnx-web/models/upscaling-stable-diffusion-x4"

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
            "output_type": "numpy",
        }
        return inputs

    def test_pipeline_default_ddim(self):
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        assert np.abs(image_slice - expected_slice).max() < 1e-1


@nightly
@require_onnxruntime
@require_torch_gpu
class OnnxStableDiffusionUpscalePipelineIntegrationTests(unittest.TestCase):
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
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(
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
