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

from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.utils.testing_utils import floats_tensor, require_peft_backend


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class LTXVideoLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = LTXPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    scheduler_kwargs = {}

    transformer_kwargs = {
        "in_channels": 8,
        "out_channels": 8,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "cross_attention_dim": 32,
        "num_layers": 1,
        "caption_channels": 32,
    }
    transformer_cls = LTXVideoTransformer3DModel
    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 8,
        "block_out_channels": (8, 8, 8, 8),
        "decoder_block_out_channels": (8, 8, 8, 8),
        "layers_per_block": (1, 1, 1, 1, 1),
        "decoder_layers_per_block": (1, 1, 1, 1, 1),
        "spatio_temporal_scaling": (True, True, False, False),
        "decoder_spatio_temporal_scaling": (True, True, False, False),
        "decoder_inject_noise": (False, False, False, False, False),
        "upsample_residual": (False, False, False, False),
        "upsample_factor": (1, 1, 1, 1),
        "timestep_conditioning": False,
        "patch_size": 1,
        "patch_size_t": 1,
        "encoder_causal": True,
        "decoder_causal": False,
    }
    vae_cls = AutoencoderKLLTXVideo
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    text_encoder_target_modules = ["q", "k", "v", "o"]

    @property
    def output_shape(self):
        return (1, 9, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 8
        num_frames = 9
        num_latent_frames = 3  # (num_frames - 1) // temporal_compression_ratio + 1
        latent_height = 8
        latent_width = 8

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels, latent_height, latent_width))
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "dance monkey",
            "num_frames": num_frames,
            "num_inference_steps": 4,
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

    @unittest.skip("Not supported in LTXVideo.")
    def test_modify_padding_mode(self):
        pass
