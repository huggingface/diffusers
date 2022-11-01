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

from diffusers import OnnxStableDiffusionImg2ImgPipeline
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
class OnnxStableDiffusionPipelineIntegrationTests(unittest.TestCase):
    def test_inference(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        provider = (
            "CUDAExecutionProvider",
            {
                "gpu_mem_limit": "17179869184",  # 16GB.
                "arena_extend_strategy": "kSameAsRequested",
            },
        )
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="onnx",
            provider=provider,
            sess_options=options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        np.random.seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=8,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.4830, 0.5242, 0.5603, 0.5016, 0.5131, 0.5111, 0.4928, 0.5025, 0.5055])
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues
        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2
