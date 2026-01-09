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
import sys
import unittest

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)

from ..testing_utils import floats_tensor, require_peft_backend


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class QwenImageLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = QwenImagePipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": 2,
        "in_channels": 16,
        "out_channels": 4,
        "num_layers": 2,
        "attention_head_dim": 16,
        "num_attention_heads": 3,
        "joint_attention_dim": 16,
        "guidance_embeds": False,
        "axes_dims_rope": (8, 4, 4),
    }
    transformer_cls = QwenImageTransformer2DModel
    z_dim = 4
    vae_kwargs = {
        "base_dim": z_dim * 6,
        "z_dim": z_dim,
        "dim_mult": [1, 2, 4],
        "num_res_blocks": 1,
        "temperal_downsample": [False, True],
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }
    vae_cls = AutoencoderKLQwenImage
    tokenizer_cls, tokenizer_id = Qwen2Tokenizer, "hf-internal-testing/tiny-random-Qwen25VLForCondGen"
    text_encoder_cls, text_encoder_id = (
        Qwen2_5_VLForConditionalGeneration,
        "hf-internal-testing/tiny-random-Qwen25VLForCondGen",
    )
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    @property
    def output_shape(self):
        return (1, 8, 8, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 8,
            "width": 8,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @unittest.skip("Not supported in Qwen Image.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Qwen Image.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Qwen Image.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Qwen Image.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Qwen Image.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Qwen Image.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Qwen Image.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Qwen Image.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass
