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

import sys
import unittest

import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils.testing_utils import (
    floats_tensor,
    require_peft_backend,
)


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class CogVideoXLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = CogVideoXPipeline
    scheduler_cls = CogVideoXDPMScheduler
    scheduler_kwargs = {"timestep_spacing": "trailing"}
    scheduler_classes = [CogVideoXDDIMScheduler, CogVideoXDPMScheduler]

    transformer_kwargs = {
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "time_embed_dim": 2,
        "text_embed_dim": 32,
        "num_layers": 1,
        "sample_width": 16,
        "sample_height": 16,
        "sample_frames": 9,
        "patch_size": 2,
        "temporal_compression_ratio": 4,
        "max_text_seq_length": 16,
    }
    transformer_cls = CogVideoXTransformer3DModel
    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        "up_block_types": (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        "block_out_channels": (8, 8, 8, 8),
        "latent_channels": 4,
        "layers_per_block": 1,
        "norm_num_groups": 2,
        "temporal_compression_ratio": 4,
    }
    vae_cls = AutoencoderKLCogVideoX
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    text_encoder_target_modules = ["q", "k", "v", "o"]

    @property
    def output_shape(self):
        return (1, 9, 16, 16, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 9
        num_latent_frames = 3  # (num_frames - 1) // temporal_compression_ratio + 1
        sizes = (2, 2)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "dance monkey",
            "num_frames": num_frames,
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
            # Cannot reduce because convolution kernel becomes bigger than sample
            "height": 16,
            "width": 16,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    @unittest.skip("Not supported in CogVideoX.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in CogVideoX.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in CogVideoX.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogVideoX.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogVideoX.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogVideoX.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogVideoX.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogVideoX.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass

    @unittest.skip("Not supported in CogVideoX.")
    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self):
        pass
