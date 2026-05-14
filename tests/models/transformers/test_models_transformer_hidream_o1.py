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
import json
import os
import sys
import tempfile
import unittest

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from diffusers import HiDreamO1Transformer2DModel  # noqa: E402
from diffusers.models.transformers.transformer_hidream_o1 import HiDreamO1AttnProcessor  # noqa: E402
from diffusers.models.transformers import transformer_hidream_o1 as hidream_o1_module  # noqa: E402

from ...testing_utils import enable_full_determinism  # noqa: E402


enable_full_determinism()


TMS_TOKEN_ID = 151673
CUDA_PARITY_ATOL = 1e-6
CUDA_PARITY_RTOL = 1e-6


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


def _get_inputs(mean=0.0, std=1.0, seed=0, device="cpu"):
    batch_size = 1
    text_seq_len = 3
    image_seq_len = 5
    total_seq_len = text_seq_len + image_seq_len
    patch_dim = 3 * 32 * 32

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vinputs = torch.randn((batch_size, image_seq_len, patch_dim), generator=generator) * std + mean

    return {
        "input_ids": torch.tensor([[11, TMS_TOKEN_ID, 17]], dtype=torch.long, device=device),
        "position_ids": torch.arange(total_seq_len, dtype=torch.long, device=device)
        .view(1, 1, -1)
        .expand(3, batch_size, -1),
        "vinputs": vinputs.to(device),
        "timestep": torch.tensor([0.25], dtype=torch.float32, device=device),
        "token_types": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long, device=device),
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


def _tensor_summary(tensor):
    tensor = tensor.detach().float()
    return {
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "min": tensor.min().item(),
        "std": tensor.std().item(),
    }


def _diff_summary(actual, expected):
    diff = (actual.detach().float() - expected.detach().float()).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


def _assert_close_with_record(actual, expected, record, key):
    record[key] = _diff_summary(actual, expected)
    try:
        torch.testing.assert_close(actual, expected, atol=CUDA_PARITY_ATOL, rtol=CUDA_PARITY_RTOL)
    except AssertionError as error:
        raise AssertionError(f"{key} mismatch with record: {json.dumps(record, sort_keys=True)}") from error


def _write_parity_report(records):
    report_path = os.environ.get("HIDREAM_O1_PARITY_REPORT")
    if not report_path:
        return

    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(records, report_file, indent=2, sort_keys=True)


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


def _sdpa_flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if key.shape[1] != query.shape[1]:
        repeat_factor = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return output.transpose(1, 2).contiguous()


class HiDreamO1Transformer2DModelTests(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "HiDream-O1 parity tests require CUDA.")
    def test_forward_uses_nonzero_zero_initialized_parameters(self):
        device = torch.device("cuda")
        model = HiDreamO1Transformer2DModel(qwen_config=_get_tiny_qwen3_vl_config().to_dict()).to(device).eval()
        _randomize_zero_parameters(model)

        with torch.no_grad():
            output_a = model(**_get_inputs(mean=0.0, std=1.0, seed=0, device=device)).sample
            output_b = model(**_get_inputs(mean=4.0, std=0.25, seed=1, device=device)).sample

        self.assertEqual(output_a.shape, (1, 8, 3072))
        self.assertGreater(output_a.abs().max().item(), 0)
        self.assertGreater((output_a - output_b).abs().max().item(), 1e-5)

    def test_attention_processor_api(self):
        model = HiDreamO1Transformer2DModel(qwen_config=_get_tiny_qwen3_vl_config().to_dict()).eval()
        processors = model.attn_processors

        self.assertEqual(len(processors), model.qwen_config.text_config.num_hidden_layers)
        self.assertTrue(all(isinstance(processor, HiDreamO1AttnProcessor) for processor in processors.values()))

        processor = HiDreamO1AttnProcessor(use_flash_attn=False)
        model.set_attn_processor(processor)
        self.assertTrue(all(attn_processor is processor for attn_processor in model.attn_processors.values()))

        model.set_default_attn_processor()
        self.assertTrue(
            all(isinstance(attn_processor, HiDreamO1AttnProcessor) for attn_processor in model.attn_processors.values())
        )

    @unittest.skipIf(not torch.cuda.is_available(), "HiDream-O1 parity tests require CUDA.")
    def test_matches_official_implementation_with_different_input_distributions(self):
        device = torch.device("cuda")
        official = _load_official_hidream_o1_module()
        official._flash_attn_func = _sdpa_flash_attn_func
        hidream_o1_module._flash_attn_func = _sdpa_flash_attn_func
        config = _get_tiny_qwen3_vl_config()

        official_model = official.Qwen3VLForConditionalGeneration(config).to(device=device, dtype=torch.bfloat16).eval()
        _randomize_zero_parameters(official_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            official_model.save_pretrained(tmpdir)
            model = HiDreamO1Transformer2DModel.from_pretrained(tmpdir).to(device=device, dtype=torch.bfloat16).eval()
            with tempfile.TemporaryDirectory() as diffusers_tmpdir:
                model.save_pretrained(diffusers_tmpdir)
                reloaded_model = (
                    HiDreamO1Transformer2DModel.from_pretrained(diffusers_tmpdir)
                    .to(device=device, dtype=torch.bfloat16)
                    .eval()
                )

                input_distributions = [
                    (0.0, 1.0, 0),
                    (3.0, 0.1, 1),
                    (-2.0, 2.5, 2),
                ]
                records = []
                previous_official_x_pred = None
                try:
                    with torch.no_grad():
                        for mean, std, seed in input_distributions:
                            inputs = _get_inputs(mean=mean, std=std, seed=seed, device=device)
                            inputs["vinputs"] = inputs["vinputs"].to(torch.bfloat16)
                            official_inputs = {**inputs, "use_flash_attn": True}
                            candidate_inputs = {**inputs, "use_flash_attn": True}
                            official_outputs = official_model.model(**official_inputs)

                            distribution_record = {
                                "cuda_device": torch.cuda.get_device_name(device),
                                "input_distribution": {
                                    "requested_mean": mean,
                                    "requested_std": std,
                                    "seed": seed,
                                    "vinputs": _tensor_summary(inputs["vinputs"]),
                                },
                                "official_x_pred": _tensor_summary(official_outputs.x_pred),
                            }
                            if previous_official_x_pred is not None:
                                distribution_record["official_x_pred_delta_from_previous"] = _diff_summary(
                                    official_outputs.x_pred, previous_official_x_pred
                                )
                            previous_official_x_pred = official_outputs.x_pred.detach().clone()

                            for candidate_name, candidate_model in (
                                ("official_checkpoint_load", model),
                                ("diffusers_reload", reloaded_model),
                            ):
                                model_outputs = candidate_model.model(**candidate_inputs)
                                wrapper_outputs = candidate_model(**candidate_inputs)
                                record = {
                                    **distribution_record,
                                    "candidate": candidate_name,
                                }
                                records.append(record)

                                _assert_close_with_record(
                                    model_outputs.last_hidden_state,
                                    official_outputs.last_hidden_state,
                                    record,
                                    "last_hidden_state",
                                )
                                _assert_close_with_record(
                                    model_outputs.x_pred,
                                    official_outputs.x_pred,
                                    record,
                                    "x_pred",
                                )
                                _assert_close_with_record(
                                    wrapper_outputs.sample,
                                    official_outputs.x_pred,
                                    record,
                                    "wrapper_sample",
                                )
                                self.assertGreater(official_outputs.x_pred.abs().max().item(), 0)
                finally:
                    _write_parity_report(records)
