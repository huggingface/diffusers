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
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.models.clip.configuration_clip import CLIPTextConfig

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.plus_models.ella import ELLA, ELLAProxyUNet
from diffusers.plus_pipelines.ella.pipeline_ella import EllaFixedDiffusionPipeline

from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_image,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import ( 
    PipelineTesterMixin,
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
)
from ...pipelines.pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)

enable_full_determinism()


class EllaDiffusionPipelineFastTests(IPAdapterTesterMixin,
                                     PipelineLatentTesterMixin,
                                     PipelineTesterMixin, 
                                     unittest.TestCase):
    print(torch_device)
    pipeline_class = EllaFixedDiffusionPipeline
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS
    params = [
        "prompt",
        "negative_prompt",
    ]
    batch_params = [
        "prompt",
        "negative_prompt",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "guidance_scale",
    ]

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            time_cond_proj_dim=None,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
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
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        
        ella = ELLA(
            time_channel=4,
            time_embed_dim=4,
            act_fn = "silu",
            out_dim = 4,
            width=32,
            layers=6,
            heads=8,
            num_latents=64,
            input_dim=2048,
        )
        
        #ella  = ELLA.from_pretrained('shauray/ELLA_SD15')

        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            norm_num_groups=2,
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        proxy_unet = ELLAProxyUNet(ella, unet)

        components = {
            "unet": unet,
            "ELLA": ella,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components


    def get_dummy_inputs(self, torch_device, seed=0):
        np.random.seed(seed)
        torch_device='cpu'
        if str(torch_device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(seed)
        inputs = {
            "prompt": "swimming underwater",
            "negative_prompt": 'bad anatomy',
            "generator": generator,
            "height": 32,
            "width": 32,
            "guidance_scale": 7.5,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_elladiffusion(self):
        torch_device = "cpu"
        components = self.get_dummy_components()

        pipe = EllaFixedDiffusionPipeline(**components)
        pipe = pipe.to(torch_device)

        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_dummy_inputs(torch_device))[0]
        image_slice = image[0, -3:, -3:, 0]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.466643, 0.6084577, 0.5677999, 0.5846181, 0.47652572, 0.5419115, 0.6090933, 0.51999027, 0.5651997])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {image_slice.flatten()}, but got {image_slice.flatten()}"
    
    @unittest.skip(reason="[to be fixed]")
    def test_ip_adapter_single(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5552, 0.5569, 0.4725, 0.4348, 0.4994, 0.4632, 0.5142, 0.5012, 0.4700])
        return super().test_ip_adapter_single(expected_pipe_slice=expected_pipe_slice)

    @unittest.skip(reason="[to be fixed]")
    def test_ip_adapter_cfg(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5552, 0.5569, 0.4725, 0.4348, 0.4994, 0.4632, 0.5142, 0.5012, 0.4700])
        return super().test_ip_adapter_cfg(expected_pipe_slice=expected_pipe_slice)

    @unittest.skip(reason="[to be fixed]")
    def test_ip_adapter_multi(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5552, 0.5569, 0.4725, 0.4348, 0.4994, 0.4632, 0.5142, 0.5012, 0.4700])
        return super().test_ip_adapter_multi(expected_pipe_slice=expected_pipe_slice)

    @unittest.skip(reason="useless")
    def test_save_load_optional_components(self):
        self.test_save_load_optional_components()
        
    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_save_load_float16(self):
        print(torch_device)
        self.test_save_load_float16()
        
    @unittest.skip(reason="useless")
    def test_save_load_local(self):
        self.test_save_load_local()
        

