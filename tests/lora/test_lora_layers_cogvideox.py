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
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils.testing_utils import (
    floats_tensor,
    require_peft_backend,
    require_torch_accelerator,
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

    def test_lora_scale_kwargs_match_fusion(self):
        super().test_lora_scale_kwargs_match_fusion(expected_atol=9e-3, expected_rtol=9e-3)

    @parameterized.expand([("block_level", True), ("leaf_level", False)])
    @require_torch_accelerator
    def test_group_offloading_inference_denoiser(self, offload_type, use_stream):
        # TODO: We don't run the (leaf_level, True) test here that is enabled for other models.
        # The reason for this can be found here: https://github.com/huggingface/diffusers/pull/11804#issuecomment-3013325338
        super()._test_group_offloading_inference_denoiser(offload_type, use_stream)

    @unittest.skip("Not supported in CogVideoX.")
    def test_modify_padding_mode(self):
        pass

    # TODO: skip them properly
