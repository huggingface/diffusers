# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
    PixArtAlphaPipeline,
    PixArtTransformer2DModel,
)

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class PixArtAlphaPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = PixArtAlphaPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    required_optional_params = PipelineTesterMixin.required_optional_params
    test_layerwise_casting = True
    test_group_offloading = True

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

    @unittest.skip("Not supported.")
    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(PVP, Sayak) need to fix later
        return

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
        expected_slice = np.array([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 0.483, 0.2583, 0.5331, 0.4852])
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

        expected_slice = np.array([0.6493, 0.537, 0.4081, 0.4762, 0.3695, 0.4711, 0.3026, 0.5218, 0.5263])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    @unittest.skip("Test is already covered through encode_prompt isolation.")
    def test_save_load_optional_components(self):
        pass

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
        expected_slice = np.array([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 0.483, 0.2583, 0.5331, 0.4852])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_raises_warning_for_mask_feature(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs.update({"mask_feature": True})

        with self.assertWarns(FutureWarning) as warning_ctx:
            _ = pipe(**inputs).images

        assert "mask_feature" in str(warning_ctx.warning)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)


@slow
@require_torch_accelerator
class PixArtAlphaPipelineIntegrationTests(unittest.TestCase):
    ckpt_id_1024 = "PixArt-alpha/PixArt-XL-2-1024-MS"
    ckpt_id_512 = "PixArt-alpha/PixArt-XL-2-512x512"
    prompt = "A small cactus with a happy face in the Sahara desert."

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_pixart_1024(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = PixArtAlphaPipeline.from_pretrained(self.ckpt_id_1024, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        image = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.0742, 0.0835, 0.2114, 0.0295, 0.0784, 0.2361, 0.1738, 0.2251, 0.3589])

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        self.assertLessEqual(max_diff, 1e-4)

    def test_pixart_512(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = PixArtAlphaPipeline.from_pretrained(self.ckpt_id_512, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)

        prompt = self.prompt

        image = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3477, 0.3882, 0.4541, 0.3413, 0.3821, 0.4463, 0.4001, 0.4409, 0.4958])

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        self.assertLessEqual(max_diff, 1e-4)

    def test_pixart_1024_without_resolution_binning(self):
        generator = torch.manual_seed(0)

        pipe = PixArtAlphaPipeline.from_pretrained(self.ckpt_id_1024, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)

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

        pipe = PixArtAlphaPipeline.from_pretrained(self.ckpt_id_512, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)

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
