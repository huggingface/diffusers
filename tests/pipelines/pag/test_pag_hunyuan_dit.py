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

import inspect
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    HunyuanDiT2DModel,
    HunyuanDiTPAGPipeline,
    HunyuanDiTPipeline,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class HunyuanDiTPAGPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = HunyuanDiTPAGPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    required_optional_params = PipelineTesterMixin.required_optional_params

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = HunyuanDiT2DModel(
            sample_size=16,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            in_channels=4,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            hidden_size=24,
            activation_fn="gelu-approximate",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDPMScheduler()
        text_encoder = BertModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "safety_checker": None,
            "feature_extractor": None,
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
            "output_type": "np",
            "use_resolution_binning": False,
            "pag_scale": 0.0,
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 16, 16, 3))
        expected_slice = np.array(
            [0.56939435, 0.34541583, 0.35915792, 0.46489206, 0.38775963, 0.45004836, 0.5957267, 0.59481275, 0.33287364]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    @unittest.skip("Not supported.")
    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(YiYi) need to fix later
        pass

    @unittest.skip("Not supported.")
    def test_sequential_offload_forward_pass_twice(self):
        # TODO(YiYi) need to fix later
        pass

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(
            expected_max_diff=1e-3,
        )

    def test_feed_forward_chunking(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_no_chunking = image[0, -3:, -3:, -1]

        pipe.transformer.enable_forward_chunking(chunk_size=1, dim=0)
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_chunking = image[0, -3:, -3:, -1]

        max_diff = np.abs(to_np(image_slice_no_chunking) - to_np(image_slice_chunking)).max()
        self.assertLess(max_diff, 1e-4)

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image = pipe(**inputs)[0]
        original_image_slice = image[0, -3:, -3:, -1]

        pipe.transformer.fuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image_fused = pipe(**inputs)[0]
        image_slice_fused = image_fused[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image_disabled = pipe(**inputs)[0]
        image_slice_disabled = image_disabled[0, -3:, -3:, -1]

        assert np.allclose(original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert np.allclose(image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert np.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline (expect same output when pag is disabled)
        pipe_sd = HunyuanDiTPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters, (
            f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__class__.__name__}."
        )
        out = pipe_sd(**inputs).images[0, -3:, -3:, -1]

        components = self.get_dummy_components()

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        # pag enabled
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 3.0
        out_pag_enabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3
        assert np.abs(out.flatten() - out_pag_enabled.flatten()).max() > 1e-3

    def test_pag_applied_layers(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        all_self_attn_layers = [k for k in pipe.transformer.attn_processors.keys() if "attn1" in k]
        original_attn_procs = pipe.transformer.attn_processors
        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # blocks.0
        block_0_self_attn = ["blocks.0.attn1.processor"]
        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0.attn1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.(0|1)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert (len(pipe.pag_attn_processors)) == 2

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0", r"blocks\.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

    @unittest.skip(
        "Test not supported as `encode_prompt` is called two times separately which deivates from about 99% of the pipelines we have."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

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
