# coding=utf-8
# Copyright 2026 chinoll and The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from diffusers import HiDreamO1ImagePipeline, HiDreamO1Transformer2DModel, UniPCMultistepScheduler  # noqa: E402

from ...testing_utils import enable_full_determinism  # noqa: E402


enable_full_determinism()

TMS_TOKEN_ID = 151673


class DummyTokenizer:
    def __init__(self):
        self.boi_token = "<|boi_token|>"
        self.bor_token = "<|bor_token|>"
        self.eor_token = "<|eor_token|>"
        self.bot_token = "<|bot_token|>"
        self.tms_token = "<|tms_token|>"

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        if return_tensors != "pt":
            raise ValueError("DummyTokenizer only supports return_tensors='pt'.")
        return torch.tensor([[11, TMS_TOKEN_ID]], dtype=torch.long)


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


def _get_tiny_qwen3_vl_config():
    text_config = Qwen3VLTextConfig(
        vocab_size=151680,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=8192,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=32,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=128,
        deepstack_visual_indexes=[],
    )
    config = Qwen3VLConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return config


def _randomize_zero_parameters(model):
    generator = torch.Generator(device="cpu").manual_seed(13)

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                continue
            if torch.count_nonzero(parameter).item() != 0:
                continue
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
            values = values.to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(values * 0.02 + 0.01)


class HiDreamO1ImagePipelineFastTests(unittest.TestCase):
    def test_text_to_image_smoke_without_vae(self):
        transformer = HiDreamO1Transformer2DModel(qwen_config=_get_tiny_qwen3_vl_config().to_dict()).eval()
        _randomize_zero_parameters(transformer)
        pipe = HiDreamO1ImagePipeline(
            processor=DummyProcessor(),
            transformer=transformer,
        )
        pipe.set_progress_bar_config(disable=True)

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe(
            "a small test prompt",
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=0.0,
            shift=1.0,
            noise_scale_start=1.0,
            noise_scale_end=1.0,
            use_resolution_binning=False,
            output_type="pt",
            generator=generator,
        ).images

        self.assertEqual(image.shape, (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(image).all())
        self.assertGreater(image.abs().max().item(), 0)

    def test_init_registers_components_with_default_scheduler(self):
        transformer = HiDreamO1Transformer2DModel(qwen_config=_get_tiny_qwen3_vl_config().to_dict()).eval()
        processor = DummyProcessor()
        pipe = HiDreamO1ImagePipeline(processor=processor, transformer=transformer)

        self.assertIs(pipe.processor, processor)
        self.assertIs(pipe.transformer, transformer)
        self.assertIsInstance(pipe.scheduler, UniPCMultistepScheduler)
