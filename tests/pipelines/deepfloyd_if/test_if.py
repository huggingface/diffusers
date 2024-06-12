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

import torch

from diffusers import (
    IFPipeline,
)
from diffusers.models.attention_processor import AttnAddedKVProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import load_numpy, require_torch_gpu, skip_mps, slow, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference
from . import IFPipelineTesterMixin


@skip_mps
class IFPipelineFastTests(PipelineTesterMixin, IFPipelineTesterMixin, unittest.TestCase):
    pipeline_class = IFPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"width", "height", "latents"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}

    def get_dummy_components(self):
        return self._get_dummy_components()

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return inputs

    def test_save_load_optional_components(self):
        self._test_save_load_optional_components()

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_save_load_float16(self):
        # Due to non-determinism in save load of the hf-internal-testing/tiny-random-t5 text encoder
        super().test_save_load_float16(expected_max_diff=1e-1)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(expected_max_diff=1e-2)

    def test_save_load_local(self):
        self._test_save_load_local()

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(
            expected_max_diff=1e-2,
        )

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=1e-3)


@slow
@require_torch_gpu
class IFPipelineSlowTests(unittest.TestCase):
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

    def test_if_text_to_image(self):
        pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        pipe.unet.set_attn_processor(AttnAddedKVProcessor())
        pipe.enable_model_cpu_offload()

        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            prompt="anime turtle",
            num_inference_steps=2,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes < 12 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if.npy"
        )
        assert_mean_pixel_difference(image, expected_image)
        pipe.remove_all_hooks()
