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
import torch

from diffusers.pipelines.stable_diffusion import OnnxStableDiffusionUpscalePipeline
from diffusers.utils import floats_tensor

from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin


class OnnxStableDiffusionUpscalePipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    # TODO: is there an appropriate internal test set?
    hub_checkpoint = "ssube/stable-diffusion-x4-upscaler-onnx"

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 128, 128), rng=random.Random(seed))
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
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

        # started as 128, should now be 512
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.6974782, 0.68902093, 0.70135885, 0.7583618, 0.7804545, 0.7854912, 0.78667426, 0.78743863, 0.78070223]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-1
