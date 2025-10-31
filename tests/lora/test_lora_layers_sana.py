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
from transformers import Gemma2Model, GemmaTokenizer

from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel

from ..testing_utils import IS_GITHUB_ACTIONS, floats_tensor, require_peft_backend


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class SanaLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = SanaPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {"shift": 7.0}
    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "num_cross_attention_heads": 2,
        "cross_attention_head_dim": 4,
        "cross_attention_dim": 8,
        "caption_channels": 8,
        "sample_size": 32,
    }
    transformer_cls = SanaTransformer2DModel
    vae_kwargs = {
        "in_channels": 3,
        "latent_channels": 4,
        "attention_head_dim": 2,
        "encoder_block_types": (
            "ResBlock",
            "EfficientViTBlock",
        ),
        "decoder_block_types": (
            "ResBlock",
            "EfficientViTBlock",
        ),
        "encoder_block_out_channels": (8, 8),
        "decoder_block_out_channels": (8, 8),
        "encoder_qkv_multiscales": ((), (5,)),
        "decoder_qkv_multiscales": ((), (5,)),
        "encoder_layers_per_block": (1, 1),
        "decoder_layers_per_block": [1, 1],
        "downsample_block_type": "conv",
        "upsample_block_type": "interpolate",
        "decoder_norm_types": "rms_norm",
        "decoder_act_fns": "silu",
        "scaling_factor": 0.41407,
    }
    vae_cls = AutoencoderDC
    tokenizer_cls, tokenizer_id = GemmaTokenizer, "hf-internal-testing/dummy-gemma"
    text_encoder_cls, text_encoder_id = Gemma2Model, "hf-internal-testing/dummy-gemma-for-diffusers"

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "",
            "negative_prompt": "",
            "num_inference_steps": 4,
            "guidance_scale": 4.5,
            "height": 32,
            "width": 32,
            "max_sequence_length": sequence_length,
            "output_type": "np",
            "complex_human_instruction": None,
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @unittest.skip("Not supported in SANA.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Not supported in SANA.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in SANA.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in SANA.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in SANA.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in SANA.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in SANA.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in SANA.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass

    @unittest.skipIf(IS_GITHUB_ACTIONS, reason="Skipping test inside GitHub Actions environment")
    def test_layerwise_casting_inference_denoiser(self):
        return super().test_layerwise_casting_inference_denoiser()
