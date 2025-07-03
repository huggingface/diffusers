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
from parameterized import parameterized
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.utils.testing_utils import floats_tensor, require_peft_backend, skip_mps


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
@skip_mps
class WanLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = WanPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 32,
        "freq_dim": 256,
        "ffn_dim": 32,
        "num_layers": 2,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 32,
    }
    transformer_cls = WanTransformer3DModel
    vae_kwargs = {
        "base_dim": 3,
        "z_dim": 16,
        "dim_mult": [1, 1, 1, 1],
        "num_res_blocks": 1,
        "temperal_downsample": [False, True, True],
    }
    vae_cls = AutoencoderKLWan
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    text_encoder_target_modules = ["q", "k", "v", "o"]

    @property
    def output_shape(self):
        return (1, 9, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 9
        num_latent_frames = 3  # (num_frames - 1) // temporal_compression_ratio + 1
        sizes = (4, 4)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "",
            "num_frames": num_frames,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @parameterized.expand([("simple",), ("weighted",), ("block_lora",), ("delete_adapter",)])
    def test_lora_set_adapters_scenarios(self, scenario):
        super()._test_lora_set_adapters_scenarios(scenario, expected_atol=9e-3)

    @parameterized.expand(
        [
            # Test actions on text_encoder LoRA only
            ("fused", "text_encoder_only"),
            ("unloaded", "text_encoder_only"),
            ("save_load", "text_encoder_only"),
            # Test actions on both text_encoder and denoiser LoRA
            ("fused", "text_and_denoiser"),
            ("unloaded", "text_and_denoiser"),
            ("unfused", "text_and_denoiser"),
            ("save_load", "text_and_denoiser"),
        ]
    )
    def test_lora_actions(self, action, components_to_add):
        super()._test_lora_actions(action, components_to_add, expected_atol=9e-3)

    @unittest.skip("Not supported in Wan.")
    def test_modify_padding_mode(self):
        pass
