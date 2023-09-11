# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, TextToVideoZeroSDXLPipeline, UNet2DConditionModel
from diffusers.utils import require_torch_gpu, slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism

enable_full_determinism()

class TextToVideoZeroSDXLPipelineFastTests(unittest.TestCase):
    def get_dummy_components(self, seed=0):
        torch.manual_seed(seed)
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
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            trained_betas=None,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="leading",
            rescale_betas_zero_snr=False,
        )
        torch.manual_seed(seed)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(seed)
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
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            # "safety_checker": None,
            # "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A panda dancing in Antarctica",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 9.0,
            "output_type": "np",
        }
        return inputs

    def get_generator(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return generator

    def test_text_to_video_zero_sdxl(self):
        components = self.get_dummy_components()
        pipe = TextToVideoZeroSDXLPipeline(**components)
        pipe = pipe.to(torch_device)

        generator = self.get_generator(torch_device)
        
        prompt = "A panda dancing in Antarctica"
        result = pipe(
            prompt=prompt,
            generator=generator,
            video_length=3,
            num_inference_steps=5,
            height=64,
            width=64,
            t0=1,
            t1=3,
            output_type="np",
        ).images

        first_frame_slice = result[0, -3:, -3:, -1]
        last_frame_slice = result[-1, -3:, -3:, 0]

        if torch_device == "cuda":
            expected_slice1 = np.array([0.50, 0.39, 0.65, 0.42, 0.24, 0.46, 0.49, 0.50, 0.47])
            expected_slice2 = np.array([0.38, 0.55, 0.69, 0.44, 0.40, 0.47, 0.57, 0.47, 0.48])
        else:
            expected_slice1 = np.array([0.37, 0.42, 0.48, 0.39, 0.43, 0.43, 0.49, 0.51, 0.54])
            expected_slice2 = np.array([0.39, 0.44, 0.48, 0.57, 0.35, 0.52, 0.68, 0.60, 0.53])

        assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 1e-2
        assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 1e-2


@slow
@require_torch_gpu
class TextToVideoZeroSDXLPipelineSlowTests(unittest.TestCase):
    def test_full_model(self):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        generator = torch.Generator(device="cuda").manual_seed(0)

        prompt = "A panda dancing in Antarctica"
        result = pipe(prompt=prompt, generator=generator).images

        first_frame_slice = result[0, -3:, -3:, -1]
        last_frame_slice = result[-1, -3:, -3:, 0]

        expected_slice1 = np.array([0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.10, 0.12, 0.12])
        expected_slice2 = np.array([0.53, 0.53, 0.53, 0.53, 0.54, 0.54, 0.53, 0.55, 0.55])

        assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 1e-2
        assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 1e-2
