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
import inspect
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusionPAGPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineFromPipeTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
)


enable_full_determinism()


class StableDiffusionPAGPipelineFastTests(
    PipelineTesterMixin,
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineFromPipeTesterMixin,
    SDXLOptionalComponentsTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionPAGPipeline
    params = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})

    def get_dummy_components(self, time_cond_proj_dim=None):
        cross_attention_dim = 8

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=2,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=cross_attention_dim,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
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
            "pag_scale": 0.9,
            "output_type": "np",
        }
        return inputs

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base  pipeline (expect same output when pag is disabled)
        pipe_sd = StableDiffusionPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert (
            "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters
        ), f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__class__.__name__}."
        out = pipe_sd(**inputs).images[0, -3:, -3:, -1]

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        # pag enabled
        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
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

        # pag_applied_layers = ["mid","up","down"] should apply to all self-attention layers
        all_self_attn_layers = [k for k in pipe.unet.attn_processors.keys() if "attn1" in k]
        original_attn_procs = pipe.unet.attn_processors
        pag_layers = [
            "down",
            "mid",
            "up",
        ]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # pag_applied_layers = ["mid"], or ["mid.block_0"] or ["mid.block_0.attentions_0"] should apply to all self-attention layers in mid_block, i.e.
        # mid_block.attentions.0.transformer_blocks.0.attn1.processor
        # mid_block.attentions.0.transformer_blocks.1.attn1.processor
        all_self_attn_mid_layers = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            # "mid_block.attentions.0.transformer_blocks.1.attn1.processor",
        ]
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block.attentions.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        # pag_applied_layers = ["mid.block_0.attentions_1"] does not exist in the model
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid_block.attentions.1"]
        with self.assertRaises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        # pag_applied_layers = "down" should apply to all self-attention layers in down_blocks
        # down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor
        # down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor
        # down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.0"]
        with self.assertRaises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.1.attentions.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 1

    def test_pag_inference(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe_pag(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (
            1,
            64,
            64,
            3,
        ), f"the shape of the output image should be (1, 64, 64, 3) but got {image.shape}"

        expected_slice = np.array(
            [0.22802538, 0.44626093, 0.48905736, 0.29633686, 0.36400637, 0.4724258, 0.4678891, 0.32260418, 0.41611585]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)


@slow
@require_torch_gpu
class StableDiffusionPAGPipelineIntegrationTests(unittest.TestCase):
    pipeline_class = StableDiffusionPAGPipeline
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", seed=1, guidance_scale=7.0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a polar bear sitting in a chair drinking a milkshake",
            "negative_prompt": "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": guidance_scale,
            "pag_scale": 3.0,
            "output_type": "np",
        }
        return inputs

    def test_pag_cfg(self):
        pipeline = AutoPipelineForText2Image.from_pretrained(self.repo_id, enable_pag=True, torch_dtype=torch.float16)
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipeline(**inputs).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)

        expected_slice = np.array(
            [0.58251953, 0.5722656, 0.5683594, 0.55029297, 0.52001953, 0.52001953, 0.49951172, 0.45410156, 0.50146484]
        )
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        ), f"output is different from expected, {image_slice.flatten()}"

    def test_pag_uncond(self):
        pipeline = AutoPipelineForText2Image.from_pretrained(self.repo_id, enable_pag=True, torch_dtype=torch.float16)
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, guidance_scale=0.0)
        image = pipeline(**inputs).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.5986328, 0.52441406, 0.3972168, 0.4741211, 0.34985352, 0.22705078, 0.4128418, 0.2866211, 0.31713867]
        )
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        ), f"output is different from expected, {image_slice.flatten()}"
