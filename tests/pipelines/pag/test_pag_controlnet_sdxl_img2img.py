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
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPAGImg2ImgPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import enable_full_determinism, floats_tensor
from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineFromPipeTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLControlNetPAGImg2ImgPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionXLControlNetPAGImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {"add_text_embeds", "add_time_ids", "add_neg_time_ids"}
    )

    # Copied from tests.pipelines.controlnet.test_controlnet_sdxl_img2img.ControlNetPipelineSDXLImg2ImgFastTests.get_dummy_components
    def get_dummy_components(self, skip_first_text_encoder=False):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        torch.manual_seed(0)
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
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }
        return components

    # based on tests.pipelines.controlnet.test_controlnet_sdxl_img2img.ControlNetPipelineSDXLImg2ImgFastTests.get_dummy_inputs
    # add `pag_scale` to the inputs
    def get_dummy_inputs(self, device, seed=0):
        controlnet_embedder_scale_factor = 2
        image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(seed),
        ).to(device)

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "output_type": "np",
            "image": image,
            "control_image": image,
        }

        return inputs

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline
        pipe_sd = StableDiffusionXLControlNetImg2ImgPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters, (
            f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__class__.__name__}."
        )
        out = pipe_sd(**inputs).images[0, -3:, -3:, -1]

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        # pag enable
        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        out_pag_enabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3
        assert np.abs(out.flatten() - out_pag_enabled.flatten()).max() > 1e-3

    def test_save_load_optional_components(self):
        pass

    def test_pag_cfg(self):
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
            [0.5562928, 0.44882968, 0.4588066, 0.63200223, 0.5694165, 0.4955688, 0.6126959, 0.57588536, 0.43827885]
        )

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        assert max_diff < 1e-3, f"output is different from expected, {image_slice.flatten()}"

    def test_pag_uncond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["guidance_scale"] = 0.0
        image = pipe_pag(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (
            1,
            64,
            64,
            3,
        ), f"the shape of the output image should be (1, 64, 64, 3) but got {image.shape}"
        expected_slice = np.array(
            [0.5543988, 0.45614323, 0.4665692, 0.6202247, 0.5598917, 0.49621183, 0.6084159, 0.5722314, 0.43945464]
        )

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        assert max_diff < 1e-3, f"output is different from expected, {image_slice.flatten()}"
