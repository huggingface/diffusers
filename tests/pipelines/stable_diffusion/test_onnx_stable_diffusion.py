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

from diffusers import OnnxStableDiffusionPipeline
from diffusers.utils.testing_utils import is_onnx_available, require_onnxruntime, require_torch_gpu, slow

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
        provider = (
            "CUDAExecutionProvider",
            {
                "gpu_mem_limit": "17179869184",  # 16GB.
                "arena_extend_strategy": "kSameAsRequested",
            },
        )
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="onnx",
            provider=provider,
            sess_options=options,
        )

        prompt = "A painting of a squirrel eating a burger"
        np.random.seed(0)
        output = sd_pipe([prompt], guidance_scale=6.0, num_inference_steps=5, output_type="np")
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3602, 0.3688, 0.3652, 0.3895, 0.3782, 0.3747, 0.3927, 0.4241, 0.4327])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.5950, -0.3039, -1.1672, 0.1594, -1.1572, 0.6719, -1.9712, -0.0403, 0.9592]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 5:
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.4776, -0.0119, -0.8519, -0.0275, -0.9764, 0.9820, -0.3843, 0.3788, 1.2264]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        test_callback_fn.has_been_called = False

        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="onnx", provider="CUDAExecutionProvider"
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "Andromeda galaxy in a bottle"

        np.random.seed(0)
        pipe(prompt=prompt, num_inference_steps=5, guidance_scale=7.5, callback=test_callback_fn, callback_steps=1)
        assert test_callback_fn.has_been_called
        assert number_of_steps == 6
