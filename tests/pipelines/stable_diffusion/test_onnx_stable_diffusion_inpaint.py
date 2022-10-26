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

from diffusers import OnnxStableDiffusionInpaintPipeline
from diffusers.utils.testing_utils import load_image, require_onnxruntime, slow

from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin


class OnnxStableDiffusionPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    # FIXME: add fast tests
    pass


@slow
@require_onnxruntime
class OnnxStableDiffusionPipelineIntegrationTests(unittest.TestCase):
    def test_stable_diffusion_inpaint_onnx(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        )

        pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision="onnx", provider="CPUExecutionProvider"
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A red cat sitting on a park bench"

        np.random.seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            guidance_scale=7.5,
            num_inference_steps=8,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 255:258, -1]

        assert images.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2951, 0.2955, 0.2922, 0.2036, 0.1977, 0.2279, 0.1716, 0.1641, 0.1799])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
