# coding=utf-8
# Copyright 2025 Harutatsu Akiyama and HuggingFace Inc.
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
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)

from ...testing_utils import assert_tensors_close, floats_tensor, torch_device
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    IPAdapterTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusionXLInstructPix2PixPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLInstructPix2PixPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
    }
    batch_input_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 5 * 8 + 32
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
            sample_size=128,
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

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device)
        image = image / 2 + 0.5
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "image_guidance_scale": 1,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestStableDiffusionXLInstructPix2PixPipeline(
    StableDiffusionXLInstructPix2PixPipelineTesterConfig, PipelineTesterMixin
):
    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` also lists the tokenizers and the text encoders, but the standard dummy inputs
        # pass a `prompt`, so those have to stay. Restrict the test to the components that can be dropped.
        droppable_components = ["image_encoder", "feature_extractor"]

        pipe = self.get_pipeline().to(torch_device)
        for optional_component in droppable_components:
            setattr(pipe, optional_component, None)

        torch.manual_seed(0)
        output = pipe(**self.get_dummy_inputs())[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in droppable_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        torch.manual_seed(0)
        output_loaded = pipe_loaded(**self.get_dummy_inputs())[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping optional components.",
        )


class TestStableDiffusionXLInstructPix2PixPipelineMemory(
    StableDiffusionXLInstructPix2PixPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL pix2pix pipeline."""


class TestStableDiffusionXLInstructPix2PixPipelineIPAdapter(
    StableDiffusionXLInstructPix2PixPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL InstructPix2Pix pipeline."""

    def _get_dummy_image_embeds(self, cross_attention_dim: int = 32):
        # Unlike the other UNet pipelines, InstructPix2Pix runs a three-way CFG batch ordered
        # [cond, negative, negative], so pre-computed embeddings carry three chunks instead of two.
        return torch.randn((3, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_faceid_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((3, 1, 1, cross_attention_dim), device=torch_device)
