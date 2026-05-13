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

import importlib.util
import os
import sys
import tempfile
import unittest

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from diffusers import HiDreamO1Transformer2DModel  # noqa: E402

from ...testing_utils import enable_full_determinism  # noqa: E402


enable_full_determinism()


TMS_TOKEN_ID = 151673


def _get_tiny_qwen3_vl_config():
    text_config = Qwen3VLTextConfig(
        vocab_size=151680,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=128,
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


def _get_inputs(mean=0.0, std=1.0, seed=0):
    batch_size = 1
    text_seq_len = 3
    image_seq_len = 5
    total_seq_len = text_seq_len + image_seq_len
    patch_dim = 3 * 32 * 32

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vinputs = torch.randn((batch_size, image_seq_len, patch_dim), generator=generator) * std + mean

    return {
        "input_ids": torch.tensor([[11, TMS_TOKEN_ID, 17]], dtype=torch.long),
        "position_ids": torch.arange(total_seq_len, dtype=torch.long).view(1, 1, -1).expand(3, batch_size, -1),
        "vinputs": vinputs,
        "timestep": torch.tensor([0.25], dtype=torch.float32),
        "token_types": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long),
        "use_flash_attn": False,
    }


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


def _load_official_hidream_o1_module():
    repo_root = os.environ.get("HIDREAM_O1_OFFICIAL_REPO", "/tmp/HiDream-O1-Image")
    module_path = os.path.join(repo_root, "models", "qwen3_vl_transformers.py")
    if not os.path.exists(module_path):
        raise unittest.SkipTest(
            "Set HIDREAM_O1_OFFICIAL_REPO or clone https://github.com/HiDream-ai/HiDream-O1-Image.git to /tmp."
        )

    spec = importlib.util.spec_from_file_location("official_hidream_o1_qwen3_vl_transformers", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HiDreamO1Transformer2DModelTests(unittest.TestCase):
    def test_forward_uses_nonzero_zero_initialized_parameters(self):
        model = HiDreamO1Transformer2DModel(qwen_config=_get_tiny_qwen3_vl_config().to_dict()).eval()
        _randomize_zero_parameters(model)

        with torch.no_grad():
            output_a = model(**_get_inputs(mean=0.0, std=1.0, seed=0)).sample
            output_b = model(**_get_inputs(mean=4.0, std=0.25, seed=1)).sample

        self.assertEqual(output_a.shape, (1, 8, 3072))
        self.assertGreater(output_a.abs().max().item(), 0)
        self.assertGreater((output_a - output_b).abs().max().item(), 1e-5)

    def test_matches_official_implementation_with_different_input_distributions(self):
        official = _load_official_hidream_o1_module()
        config = _get_tiny_qwen3_vl_config()

        official_model = official.Qwen3VLForConditionalGeneration(config).eval()
        _randomize_zero_parameters(official_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            official_model.save_pretrained(tmpdir)
            model = HiDreamO1Transformer2DModel.from_pretrained(tmpdir).eval()
            with tempfile.TemporaryDirectory() as diffusers_tmpdir:
                model.save_pretrained(diffusers_tmpdir)
                reloaded_model = HiDreamO1Transformer2DModel.from_pretrained(diffusers_tmpdir).eval()

                input_distributions = [
                    (0.0, 1.0, 0),
                    (3.0, 0.1, 1),
                    (-2.0, 2.5, 2),
                ]
                with torch.no_grad():
                    for mean, std, seed in input_distributions:
                        inputs = _get_inputs(mean=mean, std=std, seed=seed)
                        official_outputs = official_model.model(**inputs)

                        for candidate_model in (model, reloaded_model):
                            model_outputs = candidate_model.model(**inputs)
                            wrapper_outputs = candidate_model(**inputs)

                            torch.testing.assert_close(
                                model_outputs.last_hidden_state,
                                official_outputs.last_hidden_state,
                                atol=1e-6,
                                rtol=1e-6,
                            )
                            torch.testing.assert_close(
                                model_outputs.x_pred, official_outputs.x_pred, atol=1e-6, rtol=1e-6
                            )
                            torch.testing.assert_close(
                                wrapper_outputs.sample, official_outputs.x_pred, atol=1e-6, rtol=1e-6
                            )
                            self.assertGreater(official_outputs.x_pred.abs().max().item(), 0)
