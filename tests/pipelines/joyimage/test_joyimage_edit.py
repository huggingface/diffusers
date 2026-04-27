# Copyright 2025 The HuggingFace Team.
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
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    JoyImageEditPipeline,
    JoyImageEditTransformer3DModel,
)

from ...testing_utils import enable_full_determinism
from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class JoyImageEditPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = JoyImageEditPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = frozenset(["prompt", "image"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def setUp(self):
        super().setUp()
        self._bucket_patcher = patch(
            "diffusers.pipelines.joyimage.image_processor.find_best_bucket",
            return_value=(32, 32),
        )
        self._bucket_patcher.start()

    def tearDown(self):
        self._bucket_patcher.stop()
        super().tearDown()

    def get_dummy_components(self):
        tiny_ckpt_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = JoyImageEditTransformer3DModel(
            patch_size=[1, 2, 2],
            in_channels=16,
            hidden_size=32,
            num_attention_heads=2,
            text_dim=16,
            num_layers=1,
            rope_dim_list=[4, 6, 6],
            theta=256,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
                "vocab_size": 152064,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "deepstack_visual_indexes": [0],
            },
        )
        text_encoder = Qwen3VLForConditionalGeneration(config).eval()
        tokenizer = Qwen2Tokenizer.from_pretrained(tiny_ckpt_id)
        processor = Qwen3VLProcessor.from_pretrained(tiny_ckpt_id)
        processor.image_processor.min_pixels = 4 * 28 * 28
        processor.image_processor.max_pixels = 4 * 28 * 28

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "a cat sitting on a bench",
            "image": Image.new("RGB", (32, 32)),
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
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
        generated_image = image[0]

        self.assertEqual(generated_image.shape, (3, 32, 32))

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-1)

    @unittest.skip("num_images_per_prompt not applicable: each prompt is bound to a reference image")
    def test_num_images_per_prompt(self):
        pass

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @pytest.mark.xfail(condition=True, reason="Preconfigured embeddings need to be revisited.", strict=True)
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict, atol, rtol)
