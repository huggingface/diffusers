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

import gc
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, TextToVideoSDPipeline, UNet3DConditionModel
from diffusers.utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, SDFunctionTesterMixin


enable_full_determinism()


@skip_mps
class TextToVideoSDPipelineFastTests(PipelineTesterMixin, SDFunctionTesterMixin, unittest.TestCase):
    pipeline_class = TextToVideoSDPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    # No `output_type`.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet3DConditionModel(
            block_out_channels=(8, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
            cross_attention_dim=4,
            attention_head_dim=4,
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
            block_out_channels=(8,),
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=32,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=4,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
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

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pt",
        }
        return inputs

    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    def test_text_to_video_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = TextToVideoSDPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        frames = sd_pipe(**inputs).frames

        image_slice = frames[0][0][-3:, -3:, -1]
        assert frames[0][0].shape == (32, 32, 3)
        expected_slice = np.array([0.8093, 0.2751, 0.6976, 0.5927, 0.4616, 0.4336, 0.5094, 0.5683, 0.4796])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @unittest.skipIf(torch_device != "cuda", reason="Feature isn't heavily used. Test in CUDA environment only.")
    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False, expected_max_diff=3e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(test_mean_pixel_difference=False, expected_max_diff=1e-2)

    # (todo): sayakpaul
    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_consistent(self):
        pass

    # (todo): sayakpaul
    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_single_identical(self):
        pass

    @unittest.skip(reason="`num_images_per_prompt` argument is not supported for this pipeline.")
    def test_num_images_per_prompt(self):
        pass


@slow
@skip_mps
@require_torch_gpu
class TextToVideoSDPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_two_step_model(self):
        expected_video = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text-to-video/video_2step.npy"
        )

        pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
        pipe = pipe.to(torch_device)

        prompt = "Spiderman is surfing"
        generator = torch.Generator(device="cpu").manual_seed(0)

        video_frames = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").frames
        assert numpy_cosine_similarity_distance(expected_video.flatten(), video_frames.flatten()) < 1e-4

    def test_two_step_model_with_freeu(self):
        expected_video = []

        pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
        pipe = pipe.to(torch_device)

        prompt = "Spiderman is surfing"
        generator = torch.Generator(device="cpu").manual_seed(0)

        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        video_frames = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").frames
        video = video_frames[0, 0, -3:, -3:, -1].flatten()

        expected_video = [0.3643, 0.3455, 0.3831, 0.3923, 0.2978, 0.3247, 0.3278, 0.3201, 0.3475]

        assert np.abs(expected_video - video).mean() < 5e-2
