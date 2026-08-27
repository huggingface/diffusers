# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism, floats_tensor, torch_device
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..stable_diffusion.ip_adapter_tester import IPAdapterTesterMixin
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionXLControlNetImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLControlNetImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 64, 64)
    # The img2img pipeline derives its starting latents from `image`, so it takes no `latents` argument.
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {"latents"}
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {"add_text_embeds", "add_time_ids", "add_neg_time_ids"}
    )

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

        return {
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

    def get_dummy_inputs(self):
        controlnet_embedder_scale_factor = 2
        image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(0),
        ).to(torch_device)

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": image,
            "control_image": image,
        }


class TestStableDiffusionXLControlNetImg2ImgPipeline(
    StableDiffusionXLControlNetImg2ImgPipelineTesterConfig, PipelineTesterMixin
):
    # Guess mode is expected to land on the same slice at this tolerance, so both tests below share it.
    # fmt: off
    expected_slice = torch.tensor([0.55813384, 0.4668495, 0.46676695, 0.6121852, 0.55514586, 0.49157068, 0.5960574, 0.56897247, 0.43931544])
    # fmt: on

    def test_stable_diffusion_xl_controlnet_img2img(self):
        # Run on CPU: the expected slice is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        assert_tensors_close(image_slice.flatten().cpu(), self.expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_controlnet_img2img_guess(self):
        # Run on CPU: the expected slice is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["guess_mode"] = True

        image = pipe(**inputs).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # make sure that it's equal
        assert_tensors_close(image_slice.flatten().cpu(), self.expected_slice, atol=1e-2)

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


class TestStableDiffusionXLControlNetImg2ImgPipelineIPAdapter(
    StableDiffusionXLControlNetImg2ImgPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL ControlNet img2img pipeline."""


class TestStableDiffusionXLControlNetImg2ImgPipelineMemory(
    StableDiffusionXLControlNetImg2ImgPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL ControlNet img2img pipeline."""
