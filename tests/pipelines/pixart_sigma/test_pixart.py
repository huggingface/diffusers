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
    PixArtSigmaPipeline,
    PixArtTransformer2DModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
    to_np,
)


enable_full_determinism()


class PixArtSigmaPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = PixArtSigmaPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    required_optional_params = PipelineTesterMixin.required_optional_params

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = PixArtTransformer2DModel(
            sample_size=8,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
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
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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

    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(PVP, Sayak) need to fix later
        return

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
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(prompt)

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt": None,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
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
            "negative_prompt": None,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "use_resolution_binning": False,
        }

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1e-4)

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 8, 8, 3))
        expected_slice = np.array([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 0.4830, 0.2583, 0.5331, 0.4852])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_non_square_images(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs, height=32, width=48).images
        image_slice = image[0, -3:, -3:, -1]
        self.assertEqual(image.shape, (1, 32, 48, 3))

        expected_slice = np.array([0.6493, 0.5370, 0.4081, 0.4762, 0.3695, 0.4711, 0.3026, 0.5218, 0.5263])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_with_embeddings_and_multiple_images(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        prompt = inputs["prompt"]
        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        prompt_embeds, prompt_attn_mask, negative_prompt_embeds, neg_prompt_attn_mask = pipe.encode_prompt(prompt)

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attn_mask,
            "negative_prompt": None,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": neg_prompt_attn_mask,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "num_images_per_prompt": 2,
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
            "prompt_attention_mask": prompt_attn_mask,
            "negative_prompt": None,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": neg_prompt_attn_mask,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "num_images_per_prompt": 2,
            "use_resolution_binning": False,
        }

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1e-4)

    def test_inference_with_multiple_images_per_prompt(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_images_per_prompt"] = 2
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (2, 8, 8, 3))
        expected_slice = np.array([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 0.4830, 0.2583, 0.5331, 0.4852])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        original_image_slice = image[0, -3:, -3:, -1]

        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fusion_processors_exist(
            pipe.transformer
        ), "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
        assert check_qkv_fusion_matches_attn_procs_length(
            pipe.transformer, pipe.transformer.original_attn_processors
        ), "Something wrong with the attention processors concerning the fused QKV projections."

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_fused = image[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_disabled = image[0, -3:, -3:, -1]

        assert np.allclose(
            original_image_slice, image_slice_fused, atol=1e-3, rtol=1e-3
        ), "Fusion of QKV projections shouldn't affect the outputs."
        assert np.allclose(
            image_slice_fused, image_slice_disabled, atol=1e-3, rtol=1e-3
        ), "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        assert np.allclose(
            original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2
        ), "Original outputs should match when fused QKV projections are disabled."


@slow
@require_torch_gpu
class PixArtSigmaPipelineIntegrationTests(unittest.TestCase):
    ckpt_id_1024 = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    ckpt_id_512 = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
    prompt = "A small cactus with a happy face in the Sahara desert."

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_pixart_1024(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = PixArtSigmaPipeline.from_pretrained(self.ckpt_id_1024, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        prompt = self.prompt

        image = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.4517, 0.4446, 0.4375, 0.449, 0.4399, 0.4365, 0.4583, 0.4629, 0.4473])

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        self.assertLessEqual(max_diff, 1e-4)

    def test_pixart_512(self):
        generator = torch.Generator("cpu").manual_seed(0)

        transformer = PixArtTransformer2DModel.from_pretrained(
            self.ckpt_id_512, subfolder="transformer", torch_dtype=torch.float16
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            self.ckpt_id_1024, transformer=transformer, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = self.prompt

        image = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.0479, 0.0378, 0.0217, 0.0942, 0.064, 0.0791, 0.2073, 0.1975, 0.2017])

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        self.assertLessEqual(max_diff, 1e-4)

    def test_pixart_1024_without_resolution_binning(self):
        generator = torch.manual_seed(0)

        pipe = PixArtSigmaPipeline.from_pretrained(self.ckpt_id_1024, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        prompt = self.prompt
        height, width = 1024, 768
        num_inference_steps = 2

        image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1]

        generator = torch.manual_seed(0)
        no_res_bin_image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
            use_resolution_binning=False,
        ).images
        no_res_bin_image_slice = no_res_bin_image[0, -3:, -3:, -1]

        assert not np.allclose(image_slice, no_res_bin_image_slice, atol=1e-4, rtol=1e-4)

    def test_pixart_512_without_resolution_binning(self):
        generator = torch.manual_seed(0)

        transformer = PixArtTransformer2DModel.from_pretrained(
            self.ckpt_id_512, subfolder="transformer", torch_dtype=torch.float16
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            self.ckpt_id_1024, transformer=transformer, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = self.prompt
        height, width = 512, 768
        num_inference_steps = 2

        image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1]

        generator = torch.manual_seed(0)
        no_res_bin_image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
            use_resolution_binning=False,
        ).images
        no_res_bin_image_slice = no_res_bin_image[0, -3:, -3:, -1]

        assert not np.allclose(image_slice, no_res_bin_image_slice, atol=1e-4, rtol=1e-4)
