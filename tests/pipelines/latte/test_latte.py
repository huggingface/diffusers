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
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LattePipeline,
    LatteTransformer3DModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class LattPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LattePipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    required_optional_params = PipelineTesterMixin.required_optional_params

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LatteTransformer3DModel(
            sample_size=16,
            num_layers=1,
            patch_size=2,
            attention_head_dim=72,
            num_attention_heads=16,
            caption_channels=4096,
            in_channels=4,
            cross_attention_dim=1152,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDIMScheduler()
        # text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel.from_pretrained("/mnt/hwfile/gcc/maxin/work/pretrained/Latte/", subfolder="text_encoder")

        # tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("/mnt/hwfile/gcc/maxin/work/pretrained/Latte/", subfolder="tokenizer")

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder.eval(),
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 64,
            "width": 64,
            "video_length": 16,
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).video
        generated_video = video[0]

        self.assertEqual(generated_video.shape, (16, 64, 64, 3))
        expected_video = torch.randn(16, 64, 64, 3)
        max_diff = np.abs(generated_video - expected_video).max()
        # self.assertLessEqual(max_diff, 1e-3)

    def test_sequential_cpu_offload_forward_pass(self):
        pass

    def test_sequential_offload_forward_pass_twice(self):
        pass

    def test_model_cpu_offload_forward_pass(self):
        pass

    def test_save_load_float16(self):
        pass

    def test_inference_batch_single_identical(self):
        pass

    def test_float16_inference(self):
        pass

    def test_save_load_local(self):
        pass

    def test_num_images_per_prompt(self):
        pass

    def test_attention_slicing_forward_pass(self):
        pass

    def test_inference_batch_consistent(self):
        pass

    def test_cpu_offload_forward_pass_twice(self):
        pass

    def test_dict_tuple_outputs_equivalent(self):
        pass

    def test_save_load_optional_components(self):
        pass

@slow
@require_torch_gpu
class LattePipelineIntegrationTests(unittest.TestCase):
    prompt = "A painting of a squirrel eating a burger"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_latte(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = LattePipeline.from_pretrained(
            "/mnt/hwfile/gcc/maxin/work/pretrained/Latte/", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        prompt = self.prompt

        videos = pipe(
            prompt=prompt, height=512, width=512, generator=generator, num_inference_steps=2,
        ).video

        videe = videos[0]
        expected_video = torch.randn(16, 512, 512, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(videe.flatten(), expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {videe.flatten()}"