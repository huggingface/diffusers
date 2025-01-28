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

# This model implementation is heavily based on:

import inspect
import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
)
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionControlNetPAGInpaintPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionControlNetPAGInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset({"control_image"})  # skip `image` and `mask` for now, only test for control_image
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        # Copied from tests.pipelines.controlnet.test_controlnet_inpaint.ControlNetInpaintPipelineFastTests.get_dummy_components
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
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
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "controlnet": controlnet,
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

        controlnet_embedder_scale_factor = 2
        control_image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(device),
        )
        init_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        init_image = init_image.cpu().permute(0, 2, 3, 1)[0]

        image = Image.fromarray(np.uint8(init_image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(init_image + 4)).convert("RGB").resize((64, 64))

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "output_type": "np",
            "image": image,
            "mask_image": mask_image,
            "control_image": control_image,
        }

        return inputs

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base  pipeline (expect same output when pag is disabled)
        pipe_sd = StableDiffusionControlNetInpaintPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert (
            "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters
        ), f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__calss__.__name__}."
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
            [0.7488756, 0.61194265, 0.53382546, 0.5993959, 0.6193306, 0.56880975, 0.41277143, 0.5050145, 0.49376273]
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
            [0.7410303, 0.5989337, 0.530866, 0.60571927, 0.6162597, 0.5719856, 0.4187478, 0.5101238, 0.4978468]
        )

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        assert max_diff < 1e-3, f"output is different from expected, {image_slice.flatten()}"
