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

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetPAGPipeline,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    FromPipeTesterMixin,
    IPAdapterTesterMixin,
    MemoryTesterMixin,
)
from .testing_utils import PAGPipelineTesterMixin


enable_full_determinism()


class StableDiffusionXLControlNetPAGPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLControlNetPAGPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})
    output_shape = (3, 64, 64)

    def get_dummy_components(self, time_cond_proj_dim=None):
        # Copied from tests.pipelines.controlnet.test_controlnet_sdxl.StableDiffusionXLControlNetPipelineFastTests.get_dummy_components
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
            time_cond_proj_dim=time_cond_proj_dim,
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
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self):
        generator = self.get_generator(0)

        controlnet_embedder_scale_factor = 2
        # The conditioning image is drawn from the same generator, which is then handed to the pipeline in the
        # state that leaves it — the expected slices below were recorded that way.
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device("cpu"),
        )

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": image,
        }


class TestStableDiffusionXLControlNetPAGPipeline(
    StableDiffusionXLControlNetPAGPipelineTesterConfig, PAGPipelineTesterMixin
):
    base_pipeline_class = StableDiffusionXLControlNetPipeline

    def test_pag_cfg(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe_pag = self.get_pag_pipeline(pag_applied_layers=["mid", "up", "down"])

        image = pipe_pag(**self.get_dummy_inputs())[0]
        assert image.shape == (1, *self.output_shape), (
            f"the shape of the output image should be {(1, *self.output_shape)} but got {tuple(image.shape)}"
        )

        # fmt: off
        expected_slice = torch.tensor([0.6864, 0.5436, 0.5644, 0.6136, 0.5541, 0.5910, 0.4519, 0.4634, 0.5252])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_pag_uncond(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe_pag = self.get_pag_pipeline(pag_applied_layers=["mid", "up", "down"])

        inputs = self.get_dummy_inputs()
        inputs["guidance_scale"] = 0.0
        image = pipe_pag(**inputs)[0]
        assert image.shape == (1, *self.output_shape), (
            f"the shape of the output image should be {(1, *self.output_shape)} but got {tuple(image.shape)}"
        )

        # fmt: off
        expected_slice = torch.tensor([0.6843, 0.5381, 0.5675, 0.6109, 0.5493, 0.5988, 0.4477, 0.4679, 0.5242])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    @pytest.mark.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass


class TestStableDiffusionXLControlNetPAGPipelineMemory(
    StableDiffusionXLControlNetPAGPipelineTesterConfig, MemoryTesterMixin
):
    """Memory tests (CPU offload, group offload, layerwise casting) for the SDXL ControlNet PAG pipeline."""


class TestStableDiffusionXLControlNetPAGPipelineIPAdapter(
    StableDiffusionXLControlNetPAGPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL ControlNet PAG pipeline."""


class TestStableDiffusionXLControlNetPAGPipelineFromPipe(
    StableDiffusionXLControlNetPAGPipelineTesterConfig, FromPipeTesterMixin
):
    """`from_pipe` round-trip tests against `StableDiffusionXLPipeline`."""
