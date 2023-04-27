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

import random
import unittest

import torch

from diffusers import IFSuperResolutionPipeline
from diffusers.utils import floats_tensor
from diffusers.utils.testing_utils import skip_mps, torch_device

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin
from . import IFPipelineTesterMixin


@skip_mps
class IFSuperResolutionPipelineFastTests(PipelineTesterMixin, IFPipelineTesterMixin, unittest.TestCase):
    pipeline_class = IFSuperResolutionPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"width", "height"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}

    test_xformers_attention = False

    def get_dummy_components(self):
        return self._get_superresolution_dummy_components()

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "numpy",
        }

        return inputs

    def test_save_load_optional_components(self):
        self._test_save_load_optional_components()

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_save_load_float16(self):
        # Due to non-determinism in save load of the hf-internal-testing/tiny-random-t5 text encoder
        self._test_save_load_float16(expected_max_diff=1e-1)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(expected_max_diff=1e-2)

    def test_save_load_local(self):
        self._test_save_load_local()

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(
            expected_max_diff=1e-2,
        )
