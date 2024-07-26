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

import inspect
import unittest

import numpy as np
import torch

from diffusers import (
    AutoencoderKL,
    ChatGLMModel,
    ChatGLMTokenizer,
    EulerDiscreteScheduler,
    KolorsPAGPipeline,
    KolorsPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import enable_full_determinism

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineFromPipeTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class KolorsPAGPipelineFastTests(
    PipelineTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = KolorsPAGPipeline
    params = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})

    # Copied from tests.pipelines.kolors.test_kolors.KolorsPipelineFastTests.get_dummy_components
    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=56,
            cross_attention_dim=8,
            norm_num_groups=1,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder = ChatGLMModel.from_pretrained("hf-internal-testing/tiny-random-chatglm3-6b")
        tokenizer = ChatGLMTokenizer.from_pretrained("hf-internal-testing/tiny-random-chatglm3-6b")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": None,
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
            "pag_scale": 0.9,
            "output_type": "np",
        }
        return inputs

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base  pipeline (expect same output when pag is disabled)
        pipe_sd = KolorsPipeline(**components)
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
        pag_layers = ["mid", "down", "up"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        all_self_attn_mid_layers = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.1.attn1.processor",
        ]
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid.block_0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid.block_0.attentions_0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_mid_layers)

        # pag_applied_layers = ["mid.block_0.attentions_1"] does not exist in the model
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["mid.block_0.attentions_1"]
        with self.assertRaises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        # pag_applied_layers = "down" should apply to all self-attention layers in down_blocks
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 4

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down.block_0"]
        with self.assertRaises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down.block_1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 4

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down.block_1.attentions_1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

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
            [0.26030684, 0.43192005, 0.4042826, 0.4189067, 0.5181305, 0.3832534, 0.472135, 0.4145031, 0.43726248]
        )

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=3e-3)
