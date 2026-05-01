# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import os
import tempfile
from pathlib import Path

import pytest
import torch

from diffusers import AutoencoderRAE, RAEDiT2DModel
from scripts.convert_rae_stage2_to_diffusers import (
    RepoAccessor,
    build_scheduler_config,
    resolve_input_path,
    translate_transformer_state_dict,
    unwrap_state_dict,
)


def test_unwrap_state_dict_strips_supported_prefixes():
    tensor = torch.randn(1)

    assert unwrap_state_dict({"model.module.blocks.0.weight": tensor}) == {"blocks.0.weight": tensor}
    assert unwrap_state_dict({"model.blocks.0.weight": tensor}) == {"blocks.0.weight": tensor}
    assert unwrap_state_dict({"module.blocks.0.weight": tensor}) == {"blocks.0.weight": tensor}


def test_translate_transformer_state_dict_maps_feedforward_keys():
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    bias = torch.arange(4, dtype=torch.float32)
    out_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out_bias = torch.arange(2, dtype=torch.float32)

    translated = translate_transformer_state_dict(
        {
            "blocks.0.mlp.w12.weight": weight,
            "blocks.0.mlp.w12.bias": bias,
            "blocks.0.mlp.w3.weight": out_weight,
            "blocks.0.mlp.w3.bias": out_bias,
        }
    )

    assert "blocks.0.mlp.net.0.proj.weight" in translated
    assert "blocks.0.mlp.net.0.proj.bias" in translated
    assert "blocks.0.mlp.net.2.weight" in translated
    assert "blocks.0.mlp.net.2.bias" in translated
    assert torch.equal(
        translated["blocks.0.mlp.net.0.proj.weight"],
        torch.cat(weight.chunk(2, dim=0)[::-1], dim=0),
    )
    assert torch.equal(
        translated["blocks.0.mlp.net.0.proj.bias"],
        torch.cat(bias.chunk(2, dim=0)[::-1], dim=0),
    )
    assert torch.equal(translated["blocks.0.mlp.net.2.weight"], out_weight)
    assert torch.equal(translated["blocks.0.mlp.net.2.bias"], out_bias)


def test_translate_transformer_state_dict_maps_gelu_keys():
    fc1_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    fc2_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    translated = translate_transformer_state_dict(
        {
            "blocks.0.mlp.fc1.weight": fc1_weight,
            "blocks.0.mlp.fc2.weight": fc2_weight,
        }
    )

    assert torch.equal(translated["blocks.0.mlp.net.0.proj.weight"], fc1_weight)
    assert torch.equal(translated["blocks.0.mlp.net.2.weight"], fc2_weight)


def test_translate_transformer_state_dict_maps_attention_keys():
    qkv_weight = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    qkv_bias = torch.arange(12, dtype=torch.float32)
    proj_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    proj_bias = torch.arange(4, dtype=torch.float32)

    translated = translate_transformer_state_dict(
        {
            "blocks.0.attn.qkv.weight": qkv_weight,
            "blocks.0.attn.qkv.bias": qkv_bias,
            "blocks.0.attn.proj.weight": proj_weight,
            "blocks.0.attn.proj.bias": proj_bias,
        }
    )

    query_weight, key_weight, value_weight = qkv_weight.chunk(3, dim=0)
    query_bias, key_bias, value_bias = qkv_bias.chunk(3, dim=0)

    assert torch.equal(translated["blocks.0.attn.to_q.weight"], query_weight)
    assert torch.equal(translated["blocks.0.attn.to_k.weight"], key_weight)
    assert torch.equal(translated["blocks.0.attn.to_v.weight"], value_weight)
    assert torch.equal(translated["blocks.0.attn.to_q.bias"], query_bias)
    assert torch.equal(translated["blocks.0.attn.to_k.bias"], key_bias)
    assert torch.equal(translated["blocks.0.attn.to_v.bias"], value_bias)
    assert torch.equal(translated["blocks.0.attn.to_out.0.weight"], proj_weight)
    assert torch.equal(translated["blocks.0.attn.to_out.0.bias"], proj_bias)


def test_translate_transformer_state_dict_rejects_malformed_qkv_keys():
    with pytest.raises(ValueError, match="Cannot split malformed QKV tensor"):
        translate_transformer_state_dict({"blocks.0.attn.qkv.weight": torch.randn(10, 3)})


def test_translate_transformer_state_dict_preserves_pos_embed():
    pos_embed = torch.randn(1, 4, 8)

    translated = translate_transformer_state_dict({"pos_embed": pos_embed})

    assert translated["pos_embed"] is pos_embed


def test_translated_upstream_attention_keys_load_into_rae_dit():
    model = RAEDiT2DModel(
        sample_size=4,
        patch_size=1,
        in_channels=8,
        hidden_size=(32, 64),
        depth=(1, 1),
        num_heads=(4, 4),
        mlp_ratio=2.0,
        class_dropout_prob=0.0,
        num_classes=10,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    )
    upstream_state_dict = {}

    for key, value in model.state_dict().items():
        if ".attn.to_q." in key:
            prefix, suffix = key.split(".attn.to_q.")
            upstream_state_dict[f"{prefix}.attn.qkv.{suffix}"] = torch.cat(
                [
                    value,
                    model.state_dict()[f"{prefix}.attn.to_k.{suffix}"],
                    model.state_dict()[f"{prefix}.attn.to_v.{suffix}"],
                ],
                dim=0,
            )
        elif ".attn.to_k." in key or ".attn.to_v." in key:
            continue
        elif ".attn.to_out.0." in key:
            upstream_state_dict[key.replace(".attn.to_out.0.", ".attn.proj.")] = value
        elif ".attn.to_out.1." in key:
            continue
        else:
            upstream_state_dict[key] = value

    translated = translate_transformer_state_dict(upstream_state_dict)
    load_result = model.load_state_dict(translated, strict=False)
    allowed_missing = {
        "pos_embed",
        "enc_feat_rope.freqs_cos",
        "enc_feat_rope.freqs_sin",
        "dec_feat_rope.freqs_cos",
        "dec_feat_rope.freqs_sin",
    }

    assert set(load_result.missing_keys) <= allowed_missing
    assert load_result.unexpected_keys == []


def test_build_scheduler_config_rejects_non_linear_or_non_velocity_transport():
    with pytest.raises(ValueError):
        build_scheduler_config(
            {
                "transport": {"params": {"path_type": "VP", "prediction": "velocity"}},
                "misc": {"latent_size": [768, 16, 16]},
            }
        )

    with pytest.raises(ValueError):
        build_scheduler_config(
            {
                "transport": {"params": {"path_type": "Linear", "prediction": "epsilon"}},
                "misc": {"latent_size": [768, 16, 16]},
            }
        )


def test_resolve_input_path_prefers_repo_accessor_for_relative_paths():
    original_cwd = Path.cwd()

    with tempfile.TemporaryDirectory() as repo_tmpdir, tempfile.TemporaryDirectory() as cwd_tmpdir:
        repo_root = Path(repo_tmpdir)
        cwd_root = Path(cwd_tmpdir)

        repo_config = repo_root / "configs" / "sample.yaml"
        repo_config.parent.mkdir(parents=True, exist_ok=True)
        repo_config.write_text("repo: true\n", encoding="utf-8")

        cwd_config = cwd_root / "configs" / "sample.yaml"
        cwd_config.parent.mkdir(parents=True, exist_ok=True)
        cwd_config.write_text("cwd: true\n", encoding="utf-8")

        os.chdir(cwd_root)
        try:
            resolved = resolve_input_path(RepoAccessor(str(repo_root)), "configs/sample.yaml")
        finally:
            os.chdir(original_cwd)

    assert resolved == repo_config


def test_autoencoder_rae_from_pretrained_loads_local_checkpoint():
    model = AutoencoderRAE(
        encoder_type="mae",
        encoder_hidden_size=64,
        encoder_patch_size=4,
        encoder_num_hidden_layers=1,
        encoder_input_size=16,
        patch_size=4,
        image_size=16,
        num_channels=3,
        decoder_hidden_size=64,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=4,
        decoder_intermediate_size=128,
        encoder_norm_mean=[0.5, 0.5, 0.5],
        encoder_norm_std=[0.5, 0.5, 0.5],
        noise_tau=0.0,
        reshape_to_2d=True,
        scaling_factor=1.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=False)
        loaded = AutoencoderRAE.from_pretrained(tmpdir)

    assert isinstance(loaded, AutoencoderRAE)
    assert loaded.config.image_size == 16
    assert loaded.config.patch_size == 4
