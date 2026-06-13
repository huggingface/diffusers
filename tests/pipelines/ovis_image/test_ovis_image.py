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
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    OvisImagePipeline,
    OvisImageTransformer2DModel,
)

from ...testing_utils import torch_device


class OvisImagePipelineFastTests(unittest.TestCase):
    pipeline_class = OvisImagePipeline

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = OvisImageTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            axes_dims_rope=(4, 4, 8),
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
        torch.manual_seed(0)
        text_encoder = Qwen3Model(
            Qwen3Config(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=tokenizer.vocab_size + 4,
                max_position_embeddings=512,
            )
        )
        return {
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
        }

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a cat",
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "num_inference_steps": 2,
            "guidance_scale": 2.0,
            "height": 16,
            "width": 16,
            "output_type": "np",
        }

    def test_inference(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 16, 16, 3)
        assert np.isfinite(image).all()

    def test_guidance_scale_property_is_set(self):
        # The guidance_scale property reads self._guidance_scale, which __call__ must initialize.
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        pipe(**inputs)
        assert pipe.guidance_scale == inputs["guidance_scale"]

    def test_max_sequence_length_is_used(self):
        # max_sequence_length should actually bound the encoded prompt length.
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        embeds_64, _ = pipe.encode_prompt("a cat", device=torch_device, max_sequence_length=64)
        embeds_128, _ = pipe.encode_prompt("a cat", device=torch_device, max_sequence_length=128)
        assert embeds_64.shape[1] == 64
        assert embeds_128.shape[1] == 128

    def test_num_images_per_prompt(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs, num_images_per_prompt=2).images
        assert image.shape[0] == 2

    def test_batched_inference_with_default_negative_prompt(self):
        # Batched prompts with the default ("") negative prompt under CFG should not raise.
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["a cat", "a dog"]
        image = pipe(**inputs).images
        assert image.shape[0] == 2
        assert np.isfinite(image).all()
