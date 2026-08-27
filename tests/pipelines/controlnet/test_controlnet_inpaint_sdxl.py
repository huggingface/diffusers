# coding=utf-8
# Copyright 2025 Harutatsu Akiyama, Jinbin Bai, and HuggingFace Inc.
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

import random

import numpy as np
import pytest
import torch
from PIL import Image
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
    ControlNetModel,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetInpaintPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism, floats_tensor, torch_device
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionXLControlNetInpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLControlNetInpaintPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 64, 64)
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {
            "add_text_embeds",
            "add_time_ids",
            "mask",
            "masked_image_latents",
        }
    )

    def get_dummy_components(self):
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
            cross_attention_dim=64,
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

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

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

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
        }

    def get_dummy_inputs(self, img_res=64):
        # Get random floats in [0, 1] as image
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        mask_image = torch.ones_like(image)
        controlnet_embedder_scale_factor = 2
        control_image = (
            floats_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                rng=random.Random(0),
            )
            .to(torch_device)
            .cpu()
        )
        control_image = control_image.cpu().permute(0, 2, 3, 1)[0]
        # Convert image and mask_image to [0, 255]
        image = 255 * image
        mask_image = 255 * mask_image
        control_image = 255 * control_image
        # Convert to PIL image
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((img_res, img_res))
        mask_image = Image.fromarray(np.uint8(mask_image)).convert("L").resize((img_res, img_res))
        control_image = Image.fromarray(np.uint8(control_image)).convert("RGB").resize((img_res, img_res))

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": init_image,
            "mask_image": mask_image,
            "control_image": control_image,
        }


class TestStableDiffusionXLControlNetInpaintPipeline(
    StableDiffusionXLControlNetInpaintPipelineTesterConfig, PipelineTesterMixin
):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("TODO(Patrick, Sayak) - skip for now as this requires more refiner tests")
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_multi_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        # forward with single prompt
        image_slice_1 = pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = inputs["prompt"]
        image_slice_2 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "different prompt"
        image_slice_3 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        image_slice_1 = pipe(**inputs).images[0, -1, -3:, -3:]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        image_slice_2 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        image_slice_3 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

    def test_controlnet_sdxl_guess(self):
        # Run on CPU: the expected slice is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["guess_mode"] = True

        image_slice = pipe(**inputs).images[0, -1, -3:, -3:]

        # fmt: off
        expected_slice = torch.tensor([0.559845, 0.506214, 0.467439, 0.587851, 0.536138, 0.480375, 0.598892, 0.571167, 0.434700])
        # fmt: on

        # make sure that it's equal
        assert_tensors_close(image_slice.flatten().cpu(), expected_slice, atol=1e-4)


class TestStableDiffusionXLControlNetInpaintPipelineMemory(
    StableDiffusionXLControlNetInpaintPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL ControlNet inpaint pipeline."""
