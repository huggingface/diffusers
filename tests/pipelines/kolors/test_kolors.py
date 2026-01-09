# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    KolorsPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.kolors import ChatGLMModel, ChatGLMTokenizer

from ...testing_utils import enable_full_determinism
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class KolorsPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KolorsPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})

    supports_dduf = False
    test_layerwise_casting = True

    def get_dummy_components(self, time_cond_proj_dim=None):
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
            # specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=56,
            cross_attention_dim=8,
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
        text_encoder = ChatGLMModel.from_pretrained(
            "hf-internal-testing/tiny-random-chatglm3-6b", torch_dtype=torch.float32
        )
        tokenizer = ChatGLMTokenizer.from_pretrained("hf-internal-testing/tiny-random-chatglm3-6b")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": None,
            "feature_extractor": None,
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
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 64, 64, 3))
        expected_slice = np.array(
            [0.26413745, 0.4425478, 0.4102801, 0.42693347, 0.52529025, 0.3867405, 0.47512037, 0.41538602, 0.43855375]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=2e-4)

    def test_save_load_float16(self):
        super().test_save_load_float16(expected_max_diff=2e-1)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=5e-3)
