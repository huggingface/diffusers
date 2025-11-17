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
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from diffusers import (
    Flux2Pipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLFlux2
)

from ..testing_utils import floats_tensor, require_peft_backend
   

sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class Flux2LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = Flux2Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 16,
        "timestep_guidance_channels": 256,
        "axes_dims_rope": [4, 4, 4, 4],
    }
    transformer_cls = Flux2Transformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",),
        "up_block_types": ("UpDecoderBlock2D",),
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 1,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }
    vae_cls = AutoencoderKLFlux2

    tokenizer_cls, tokenizer_id = AutoProcessor, "hf-internal-testing/tiny-mistral3-diffusers"
    text_encoder_cls, text_encoder_id = Mistral3ForConditionalGeneration, "hf-internal-testing/tiny-mistral3-diffusers"
    denoiser_target_modules = ["to_qkv_mlp_proj", "to_k"]

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
            "prompt": "a dog is dancing",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 8,
            "output_type": "np",
            "text_encoder_out_layers": (1,)
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @unittest.skip("Not supported in Flux2.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Flux2.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Flux2.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Flux2.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Flux2.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Flux2.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Flux2.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Flux2.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass
