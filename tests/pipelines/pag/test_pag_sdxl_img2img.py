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
import inspect
import random
import unittest

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineFromPipeTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLPAGImg2ImgPipelineFastTests(
    PipelineTesterMixin,
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionXLPAGImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS.union({"pag_scale", "pag_adaptive_scale"}) - {"height", "width"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {"add_text_embeds", "add_time_ids", "add_neg_time_ids"}
    )

    supports_dduf = False

    #  based on tests.pipelines.stable_diffusion_xl.test_stable_diffusion_xl_img2img_pipeline.get_dummy_components
    def get_dummy_components(
        self, skip_first_text_encoder=False, time_cond_proj_dim=None, requires_aesthetics_score=False
    ):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            time_cond_proj_dim=time_cond_proj_dim,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=72 if requires_aesthetics_score else 80,  # 5 * 8 + 32
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
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
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            image_size=224,
            projection_dim=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )

        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        feature_extractor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
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
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "requires_aesthetics_score": requires_aesthetics_score,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
        }
        return components

    # based on tests.pipelines.stable_diffusion_xl.test_stable_diffusion_xl_img2img_pipeline.StableDiffusionXLImg2ImgPipelineFastTests
    # add `pag_scale` to the inputs
    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "pag_scale": 3.0,
            "output_type": "np",
            "strength": 0.8,
        }
        return inputs

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(requires_aesthetics_score=True)

        # base pipeline
        pipe_sd = StableDiffusionXLImg2ImgPipeline(**components)
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

        # pag enabled
        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        out_pag_enabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3
        assert np.abs(out.flatten() - out_pag_enabled.flatten()).max() > 1e-3

    def test_pag_inference(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(requires_aesthetics_score=True)

        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe_pag(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (
            1,
            32,
            32,
            3,
        ), f"the shape of the output image should be (1, 64, 64, 3) but got {image.shape}"
        expected_slice = np.array([0.4613, 0.4902, 0.4406, 0.6788, 0.5611, 0.4529, 0.5893, 0.5975, 0.5226])

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        assert max_diff < 1e-3, f"output is different from expected, {image_slice.flatten()}"

    @unittest.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass


@slow
@require_torch_accelerator
class StableDiffusionXLPAGImg2ImgPipelineIntegrationTests(unittest.TestCase):
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", seed=0, guidance_scale=7.0):
        img_url = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
        )

        init_image = load_image(img_url)

        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a dog catching a frisbee in the jungle",
            "generator": generator,
            "image": init_image,
            "strength": 0.8,
            "num_inference_steps": 3,
            "guidance_scale": guidance_scale,
            "pag_scale": 3.0,
            "output_type": "np",
        }
        return inputs

    def test_pag_cfg(self):
        pipeline = AutoPipelineForImage2Image.from_pretrained(self.repo_id, enable_pag=True, torch_dtype=torch.float16)
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipeline(**inputs).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        expected_slice = np.array(
            [0.20301354, 0.21078318, 0.2021082, 0.20277798, 0.20681083, 0.19562206, 0.20121682, 0.21562952, 0.21277016]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3, (
            f"output is different from expected, {image_slice.flatten()}"
        )

    def test_pag_uncond(self):
        pipeline = AutoPipelineForImage2Image.from_pretrained(self.repo_id, enable_pag=True, torch_dtype=torch.float16)
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, guidance_scale=0.0)
        image = pipeline(**inputs).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        expected_slice = np.array(
            [0.21303111, 0.22188407, 0.2124992, 0.21365267, 0.18823743, 0.17569828, 0.21113116, 0.19419771, 0.18919235]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3, (
            f"output is different from expected, {image_slice.flatten()}"
        )
