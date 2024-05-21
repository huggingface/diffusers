# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
import gc
import os
import random
import unittest
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    LCMScheduler,
    MarigoldDepthPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    slow,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class MarigoldDepthPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = MarigoldDepthPipeline
    params = frozenset(["image"])
    batch_params = frozenset(["image"])
    image_params = frozenset(["image"])
    image_latents_params = frozenset(["latents"])
    callback_cfg_params = frozenset([])
    test_xformers_attention = False
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "output_type",
        ]
    )

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        torch.manual_seed(0)
        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
            beta_schedule="scaled_linear",
            clip_sample=False,
            thresholding=False,
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
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image": image,
            "num_inference_steps": 1,
            "processing_resolution": 0,
            "generator": generator,
            "output_type": "np",
        }
        return inputs
    
    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)



@slow
@require_torch_gpu
class MarigoldDepthPipelineIntegrationTests(unittest.TestCase):
    save_output: bool = True
    enable_asserts: bool = True
    model_id: str = "prs-eth/marigold-lcm-v1-0"
    url_input_basedir: str = "https://marigoldmonodepth.github.io/images"
    url_output_basedir: str = (
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/marigold"
    )
    progress_bar_kwargs = {"disable": True}

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def _test_marigold_depth(
        self,
        is_fp16: bool = True,
        device: str = "cuda",
        enable_model_cpu_offload: bool = False,
        generator_seed: int = 0,
        num_inference_steps: int = 1,
        processing_resolution: int = 768,
        ensemble_size: int = 1,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        match_input_resolution: bool = True,
        fname_input: str = "einstein.jpg",
        atol: float = 1e-4,
    ):
        out_basename = os.path.splitext(os.path.basename(fname_input))[0]
        out_basename += "_" + self.model_id.replace("/", "-").replace("_", "-")

        from_pretrained_kwargs = {}
        if is_fp16:
            from_pretrained_kwargs["variant"] = "fp16"
            from_pretrained_kwargs["torch_dtype"] = torch.float16
            out_basename += "_f16"
        else:
            out_basename += "_f32"

        pipe = MarigoldDepthPipeline.from_pretrained(self.model_id, **from_pretrained_kwargs)

        if isinstance(self.progress_bar_kwargs, dict):
            pipe.set_progress_bar_config(**self.progress_bar_kwargs)

        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
            out_basename += "_cpuoffl"
        else:
            pipe.to(device)
            out_basename += "_" + str(device)

        generator = torch.Generator(device=device).manual_seed(generator_seed)
        out_basename += f"_G{generator_seed}"

        image_url = f"{self.url_input_basedir}/{fname_input}"
        image = load_image(image_url)
        width, height = image.size

        out = pipe(
            image,
            num_inference_steps=num_inference_steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_resolution,
            match_input_resolution=match_input_resolution,
            resample_method_input="bilinear",
            resample_method_output="bilinear",
            batch_size=batch_size,
            ensembling_kwargs=ensembling_kwargs,
            latents=None,
            generator=generator,
            output_type="np",
            output_uncertainty=False,
            output_latent=False,
        )

        out_basename += f"_S{num_inference_steps}"
        out_basename += f"_P{processing_resolution}"
        out_basename += f"_E{ensemble_size}"
        out_basename += f"_B{batch_size}"
        out_basename += f"_M{int(match_input_resolution)}"

        expected_image_fname = f"{out_basename}.png"
        expected_image_url = f"{self.url_output_basedir}/{expected_image_fname}"

        vis = pipe.image_processor.visualize_depth(out.prediction)[0]
        if self.save_output:
            vis.save(expected_image_fname)

        # No asserts above this line!

        if not self.enable_asserts:
            return

        if match_input_resolution:
            self.assertEqual(out.prediction.shape[2:], (height, width), "Unexpected output resolution")
        else:
            self.assertEqual(max(out.prediction.shape[2:]), processing_resolution, "Unexpected output resolution")

        expected_image = np.array(load_image(expected_image_url))
        vis = np.array(vis)
        self.assertTrue(np.allclose(vis, expected_image, atol=atol))

    def test_marigold_depth_einstein_f32_cpu_G0_S1_P32_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=False,
            device="cpu",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=32,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f32_cuda_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=False,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f32_cpuoffl_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=False,
            device="cuda",
            enable_model_cpu_offload=True,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cpuoffl_G0_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=True,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G2024_S1_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=2024,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S2_P768_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=2,
            processing_resolution=768,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P512_E1_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P768_E3_B1_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=3,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=1,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P768_E4_B2_M1(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=768,
            ensemble_size=4,
            ensembling_kwargs={"reduction": "mean"},
            batch_size=2,
            match_input_resolution=True,
            fname_input="einstein.jpg",
        )

    def test_marigold_depth_einstein_f16_cuda_G0_S1_P512_E1_B1_M0(self):
        self._test_marigold_depth(
            is_fp16=True,
            device="cuda",
            enable_model_cpu_offload=False,
            generator_seed=0,
            num_inference_steps=1,
            processing_resolution=512,
            ensemble_size=1,
            batch_size=1,
            match_input_resolution=False,
            fname_input="einstein.jpg",
        )
