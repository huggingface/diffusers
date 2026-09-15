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

import gc

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    EulerDiscreteScheduler,
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
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


class StableDiffusionXLPAGPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLPAGPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"})
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})
    output_shape = (3, 64, 64)

    def get_dummy_components(self, time_cond_proj_dim=None):
        # Copied from tests.pipelines.stable_diffusion_xl.test_stable_diffusion_xl.StableDiffusionXLPipelineTesterConfig.get_dummy_components
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
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
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
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "pag_scale": 0.9,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableDiffusionXLPAGPipeline(StableDiffusionXLPAGPipelineTesterConfig, PAGPipelineTesterMixin):
    base_pipeline_class = StableDiffusionXLPipeline
    # fmt: off
    expected_pag_slice = torch.tensor([0.5565, 0.5305, 0.4652, 0.4330, 0.4823, 0.4640, 0.5191, 0.4983, 0.4684])
    # fmt: on

    def test_pag_applied_layers(self):
        pipe = self.get_pipeline()

        # pag_applied_layers = ["mid","up","down"] should apply to all self-attention layers
        all_self_attn_layers = [k for k in pipe.unet.attn_processors.keys() if "attn1" in k]
        original_attn_procs = pipe.unet.attn_processors
        pag_layers = ["mid", "down", "up"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # pag_applied_layers = ["mid"], or ["mid.block_0"] or ["mid.block_0.attentions_0"] should apply to all self-attention layers in mid_block, i.e.
        # mid_block.attentions.0.transformer_blocks.0.attn1.processor
        # mid_block.attentions.0.transformer_blocks.1.attn1.processor
        all_self_attn_mid_layers = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
            "mid_block.attentions.0.transformer_blocks.1.attn1.processor",
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
        with pytest.raises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        # pag_applied_layers = "down" should apply to all self-attention layers in down_blocks
        # down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor
        # down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor
        # down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor
        # down_blocks.1.attentions.1.transformer_blocks.1.attn1.processor
        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 4

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.0"]
        with pytest.raises(ValueError):
            pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 4

        pipe.unet.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["down_blocks.1.attentions.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

    @pytest.mark.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass


class TestStableDiffusionXLPAGPipelineMemory(StableDiffusionXLPAGPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL PAG pipeline."""


class TestStableDiffusionXLPAGPipelineIPAdapter(StableDiffusionXLPAGPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the SDXL PAG pipeline."""


class TestStableDiffusionXLPAGPipelineFromPipe(StableDiffusionXLPAGPipelineTesterConfig, FromPipeTesterMixin):
    """`from_pipe` round-trip tests against `StableDiffusionXLPipeline`."""


@slow
@require_torch_accelerator
class TestStableDiffusionXLPAGPipelineIntegration:
    pipeline_class = StableDiffusionXLPAGPipeline
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", seed=0, guidance_scale=7.0):
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
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipeline(**inputs).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        # fmt: off
        expected_slice = Expectations(
            {
                ("cuda", 7): np.array([0.68121016, 0.709187, 0.7076755, 0.6825691, 0.6788969, 0.6774048, 0.6794217, 0.6880745 , 0.6896945]),
                ("xpu", None): np.array([0.67993283, 0.7091448, 0.70783514, 0.6810594, 0.6776446, 0.6758386, 0.6778576, 0.6865928 , 0.6879399]),
            }
        ).get_expectation()
        # fmt: on
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3, (
            f"output is different from expected, {image_slice.flatten()}"
        )

    def test_pag_uncond(self):
        pipeline = AutoPipelineForText2Image.from_pretrained(self.repo_id, enable_pag=True, torch_dtype=torch.float16)
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, guidance_scale=0.0)
        image = pipeline(**inputs).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        # fmt: off
        expected_slice = Expectations(
            {
                ("cuda", 7): np.array([0.5665728, 0.58520794, 0.57458067, 0.4698207, 0.48426265, 0.48946208, 0.46729165, 0.47325698, 0.43762192]),
                ("xpu", 5): np.array([0.565891, 0.5846026, 0.5738897, 0.46956095, 0.48404077, 0.48921135, 0.4662642, 0.4720734, 0.43678787]),
                ("xpu", 3): np.array([0.5658767, 0.58458394, 0.57346797, 0.46756423, 0.48198313, 0.48735905, 0.46382266, 0.46961138, 0.43374598]),
            }
        ).get_expectation()
        # fmt: on
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3, (
            f"output is different from expected, {image_slice.flatten()}"
        )
