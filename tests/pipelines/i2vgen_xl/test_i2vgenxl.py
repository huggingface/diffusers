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

import random
import unittest

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    I2VGenXLPipeline,
)
from diffusers.models.unets import I2VGenXLUNet
from diffusers.utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    print_tensor_test,
    skip_mps,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


@skip_mps
class I2VGenPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = I2VGenXLPipeline
    params = frozenset(["prompt", "negative_prompt", "image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "generator"])
    # No `output_type`.
    required_optional_params = frozenset(["num_inference_steps", "generator", "latents", "return_dict"])

    def get_dummy_components(self):
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        torch.manual_seed(0)
        unet = I2VGenXLUNet(
            block_out_channels=(4, 8),
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

        torch.manual_seed(0)
        vision_encoder_config = CLIPVisionConfig(
            hidden_size=4,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            intermediate_size=16,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(vision_encoder_config)

        torch.manual_seed(0)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "image_encoder": image_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": input_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pt",
            "num_frames": 4,
            "width": 32,
            "height": 32,
        }
        return inputs

    def test_text_to_video_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        frames = pipe(**inputs).frames

        image_slice = frames[0][0][-3:, -3:, -1]

        assert frames[0][0].shape == (32, 32, 3)
        expected_slice = np.array([0.5146, 0.6525, 0.6032, 0.5204, 0.5675, 0.4125, 0.3016, 0.5172, 0.4095])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_save_load_local(self):
        super().test_save_load_local(expected_max_difference=0.006)

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=0.008)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=0.004)

    @unittest.skip("Deprecated functionality")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(test_mean_pixel_difference=False, expected_max_diff=1e-2)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(batch_size=2, expected_max_diff=0.008)

    def test_num_videos_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        frames = pipe(**inputs, num_videos_per_prompt=2).frames

        assert frames.shape == (2, 4, 32, 32, 3)
        assert frames[0][0].shape == (32, 32, 3)

        image_slice = frames[0][0][-3:, -3:, -1]
        expected_slice = np.array([0.5146, 0.6525, 0.6032, 0.5204, 0.5675, 0.4125, 0.3016, 0.5172, 0.4095])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
