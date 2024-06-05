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
            sample_size=64,
            num_layers=28,
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
            "text_encoder": text_encoder,
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
            "use_resolution_binning": False,
            "output_type": "np",
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
        video_slice = video[0]

        self.assertEqual(video_slice.shape, (16, 512, 512, 3))
        expected_video = torch.randn(16, 512, 512, 3).numpy()
        max_diff = np.abs(video_slice.flatten() - expected_video).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(Xin Ma) need to fix later
        pass

    def test_sequential_offload_forward_pass_twice(self):
        # TODO(Xin Ma) need to fix later
        pass

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(
            expected_max_diff=1e-3,
        )

    def test_save_load_optional_components(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        prompt = inputs["prompt"]
        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(prompt, device=torch_device, dtype=torch.float32, text_encoder_index=0)

        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = pipe.encode_prompt(
            prompt,
            device=torch_device,
            dtype=torch.float32,
            text_encoder_index=1,
        )

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_attention_mask_2": prompt_attention_mask_2,
            "negative_prompt_embeds_2": negative_prompt_embeds_2,
            "negative_prompt_attention_mask_2": negative_prompt_attention_mask_2,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "use_resolution_binning": False,
        }

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(torch_device)

        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_attention_mask_2": prompt_attention_mask_2,
            "negative_prompt_embeds_2": negative_prompt_embeds_2,
            "negative_prompt_attention_mask_2": negative_prompt_attention_mask_2,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "use_resolution_binning": False,
        }

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1e-4)


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