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

import numpy as np
import pytest
import torch
from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast

from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils.testing_utils import (
    floats_tensor,
    is_peft_available,
    is_torch_version,
    require_peft_backend,
    skip_mps,
    torch_device,
)


if is_peft_available():
    pass

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
@skip_mps
class HunyuanVideoLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = HunyuanVideoPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    scheduler_kwargs = {}

    transformer_kwargs = {
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_refiner_layers": 1,
        "pooled_projection_dim": 768,
        "text_embed_dim": 4096,
        "in_channels": 16,
        "mlp_ratio": 1.0,
        "out_channels": 16,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
    }
    transformer_cls = HunyuanVideoTransformer3DModel
    vae_kwargs = {
        "act_fn": "silu",
        "down_block_types": (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        "up_block_types": (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 16,
        "layers_per_block": 1,
        "norm_num_groups": 1,
        "scaling_factor": 0.476986,
        "spatial_compression_ratio": 8,
        "temporal_compression_ratio": 4,
        "block_out_channels": (1, 1, 1, 1),
    }
    vae_cls = AutoencoderKLHunyuanVideo
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = LlamaTokenizerFast, "HunyuanVideo/tokenizer"
    tokenizer_2_cls, tokenizer_2_id = CLIPTokenizer, "HunyuanVideo/tokenizer_2"
    text_encoder_cls, text_encoder_id = LlamaModel, "HunyuanVideo/text_encoder"
    text_encoder_2_cls, text_encoder_2_id = CLIPTextModel, "HunyuanVideo/text_encoder_2"

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
            "num_inference_steps": 1,
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

    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cpu" and is_torch_version(">=", "2.5"),
        reason="Test currently fails on CPU and PyTorch 2.5.1 but not on PyTorch 2.4.1.",
        strict=True,
    )
    def test_lora_fuse_nan(self):
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")

            self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

            # corrupt one LoRA weight with `inf` values
            with torch.no_grad():
                pipe.transformer.transformer_blocks[0].attn.to_q.lora_A["adapter-1"].weight += float("inf")

            # with `safe_fusing=True` we should see an Error
            with self.assertRaises(ValueError):
                pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

            # without we should not see an error, but every image will be black
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)

            out = pipe(
                prompt=inputs["prompt"], height=inputs["height"], width=inputs["width"], num_frames=inputs["num_frames"], num_inference_steps=inputs["num_inference_steps"], max_sequence_length=inputs["max_sequence_length"], output_type="np"
            )[0]

            self.assertTrue(np.isnan(out).all())

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    @unittest.skip("Not supported in HunyuanVideo.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in HunyuanVideo.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in HunyuanVideo.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass