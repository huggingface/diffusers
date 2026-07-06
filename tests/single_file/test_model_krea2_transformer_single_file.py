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

"""
Tests for Krea2Transformer2DModel from_single_file() support.

Two suites:
  1. TestKrea2ConversionFunction  — offline unit tests for the key-mapping
     function. No checkpoint, no GPU, no network.
  2. TestKrea2TransformerSingleFile — @nightly integration test stub for when
     a public single-file checkpoint becomes available.
"""

import unittest

import torch

from diffusers import Krea2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers

from ..testing_utils import enable_full_determinism
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


# ---------------------------------------------------------------------------
# Helper: build a synthetic "original" Krea2 state dict
# ---------------------------------------------------------------------------

def _make_krea2_original_state_dict(
    num_layers=2,
    hidden_size=32,      # num_heads * head_dim
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=32,
    timestep_embed_dim=8,
    text_hidden_dim=16,
    num_text_layers=3,
    text_num_attention_heads=2,
    text_num_key_value_heads=1,
    text_intermediate_size=16,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    in_channels=16,
    prefix="diffusion_model.",
):
    """
    Produces a synthetic state dict in the pre-diffusers Krea2 format:
      blocks.{i}.attn.{wq,wk,wv,wo,gate}
      blocks.{i}.mlp.{gate,up,down}
      blocks.{i}.{norm1,norm2,scale_shift_table}
      txtfusion.{layerwise_blocks,refiner_blocks}.{i}.attn.*
      txtfusion.{layerwise_blocks,refiner_blocks}.{i}.mlp.*
      txtfusion.projector
      img_in, time_embed.*, time_mod_proj, txt_in.*, final_layer.*, rotary_emb.*
    """
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    sd = {}

    def p(name):
        return f"{prefix}{name}"

    # head_dim for text fusion attention (uses text_num_attention_heads, not num_heads)
    text_head_dim = hidden_size // text_num_attention_heads

    # Top-level pass-through modules
    sd[p("img_in.weight")] = torch.randn(hidden_size, in_channels)
    sd[p("img_in.bias")] = torch.randn(hidden_size)
    sd[p("time_embed.linear_1.weight")] = torch.randn(hidden_size, timestep_embed_dim)
    sd[p("time_embed.linear_1.bias")] = torch.randn(hidden_size)
    sd[p("time_embed.linear_2.weight")] = torch.randn(hidden_size, hidden_size)
    sd[p("time_embed.linear_2.bias")] = torch.randn(hidden_size)
    sd[p("time_mod_proj.weight")] = torch.randn(6 * hidden_size, hidden_size)
    sd[p("time_mod_proj.bias")] = torch.randn(6 * hidden_size)
    sd[p("txt_in.norm.weight")] = torch.randn(text_hidden_dim)
    sd[p("txt_in.linear_1.weight")] = torch.randn(hidden_size, text_hidden_dim)
    sd[p("txt_in.linear_1.bias")] = torch.randn(hidden_size)
    sd[p("txt_in.linear_2.weight")] = torch.randn(hidden_size, hidden_size)
    sd[p("txt_in.linear_2.bias")] = torch.randn(hidden_size)
    # rotary_emb (Krea2RotaryPosEmbed) computes frequencies on the fly — no state dict entries

    # Transformer blocks (original naming: blocks.{i})
    for i in range(num_layers):
        b = p(f"blocks.{i}.")
        sd[f"{b}attn.wq.weight"] = torch.randn(hidden_size, hidden_size)
        sd[f"{b}attn.wk.weight"] = torch.randn(kv_dim, hidden_size)
        sd[f"{b}attn.wv.weight"] = torch.randn(kv_dim, hidden_size)
        sd[f"{b}attn.wo.weight"] = torch.randn(hidden_size, hidden_size)
        sd[f"{b}attn.gate.weight"] = torch.randn(hidden_size, hidden_size)
        # norm_q/norm_k: same names in both formats, pass through via the block renaming
        sd[f"{b}attn.norm_q.weight"] = torch.randn(head_dim)
        sd[f"{b}attn.norm_k.weight"] = torch.randn(head_dim)
        sd[f"{b}mlp.gate.weight"] = torch.randn(intermediate_size, hidden_size)
        sd[f"{b}mlp.up.weight"] = torch.randn(intermediate_size, hidden_size)
        sd[f"{b}mlp.down.weight"] = torch.randn(hidden_size, intermediate_size)
        sd[f"{b}norm1.weight"] = torch.randn(hidden_size)
        sd[f"{b}norm2.weight"] = torch.randn(hidden_size)
        sd[f"{b}scale_shift_table"] = torch.randn(6, hidden_size)

    # Text fusion blocks (original naming: txtfusion.*)
    for group, n_blocks in (("layerwise_blocks", num_layerwise_text_blocks),
                             ("refiner_blocks", num_refiner_text_blocks)):
        for i in range(n_blocks):
            b = p(f"txtfusion.{group}.{i}.")
            sd[f"{b}attn.wq.weight"] = torch.randn(hidden_size, hidden_size)
            sd[f"{b}attn.wk.weight"] = torch.randn(hidden_size, hidden_size)
            sd[f"{b}attn.wv.weight"] = torch.randn(hidden_size, hidden_size)
            sd[f"{b}attn.wo.weight"] = torch.randn(hidden_size, hidden_size)
            sd[f"{b}attn.gate.weight"] = torch.randn(hidden_size, hidden_size)
            # norm_q/norm_k in text fusion attention use text_head_dim
            sd[f"{b}attn.norm_q.weight"] = torch.randn(text_head_dim)
            sd[f"{b}attn.norm_k.weight"] = torch.randn(text_head_dim)
            sd[f"{b}mlp.gate.weight"] = torch.randn(text_intermediate_size, hidden_size)
            sd[f"{b}mlp.up.weight"] = torch.randn(text_intermediate_size, hidden_size)
            sd[f"{b}mlp.down.weight"] = torch.randn(hidden_size, text_intermediate_size)
            sd[f"{b}norm1.weight"] = torch.randn(hidden_size)
            sd[f"{b}norm2.weight"] = torch.randn(hidden_size)

    # txtfusion.projector (top-level, maps to text_fusion.projector, no bias)
    sd[p("txtfusion.projector.weight")] = torch.randn(1, num_text_layers)

    # Final layer (pass-through)
    sd[p("final_layer.norm.weight")] = torch.randn(hidden_size)
    sd[p("final_layer.linear.weight")] = torch.randn(in_channels, hidden_size)
    sd[p("final_layer.linear.bias")] = torch.randn(in_channels)
    sd[p("final_layer.scale_shift_table")] = torch.randn(2, hidden_size)

    return sd


class TestKrea2ConversionFunction(unittest.TestCase):
    """Offline unit tests for convert_krea2_transformer_checkpoint_to_diffusers."""

    def _convert(self, prefix="diffusion_model.", num_layers=2):
        sd = _make_krea2_original_state_dict(prefix=prefix, num_layers=num_layers)
        return convert_krea2_transformer_checkpoint_to_diffusers(sd)

    # ------------------------------------------------------------------
    # Prefix stripping
    # ------------------------------------------------------------------

    def test_strips_diffusion_model_prefix(self):
        result = self._convert(prefix="diffusion_model.")
        self.assertFalse(any(k.startswith("diffusion_model.") for k in result))

    def test_no_prefix_passthrough(self):
        sd = _make_krea2_original_state_dict(prefix="")
        result = convert_krea2_transformer_checkpoint_to_diffusers(sd)
        self.assertIn("transformer_blocks.0.attn.to_q.weight", result)

    # ------------------------------------------------------------------
    # Transformer block renaming: blocks.{i} -> transformer_blocks.{i}
    # ------------------------------------------------------------------

    def test_transformer_block_attention_keys(self):
        result = self._convert()
        expected = {
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.attn.to_gate.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.norm_k.weight",
        }
        for key in expected:
            self.assertIn(key, result, msg=f"Missing key: {key}")

    def test_original_attn_keys_absent(self):
        result = self._convert()
        for i in range(2):
            self.assertNotIn(f"blocks.{i}.attn.wq.weight", result)
            self.assertNotIn(f"blocks.{i}.attn.wo.weight", result)

    def test_transformer_block_ff_keys(self):
        result = self._convert()
        for i in range(2):
            self.assertIn(f"transformer_blocks.{i}.ff.gate.weight", result)
            self.assertIn(f"transformer_blocks.{i}.ff.up.weight", result)
            self.assertIn(f"transformer_blocks.{i}.ff.down.weight", result)

    def test_original_mlp_keys_absent(self):
        result = self._convert()
        for i in range(2):
            self.assertNotIn(f"blocks.{i}.mlp.gate.weight", result)

    def test_transformer_block_norm_and_scale_shift_keys(self):
        result = self._convert()
        for i in range(2):
            self.assertIn(f"transformer_blocks.{i}.norm1.weight", result)
            self.assertIn(f"transformer_blocks.{i}.norm2.weight", result)
            self.assertIn(f"transformer_blocks.{i}.scale_shift_table", result)

    # ------------------------------------------------------------------
    # Text fusion renaming: txtfusion -> text_fusion
    # ------------------------------------------------------------------

    def test_text_fusion_attention_keys(self):
        result = self._convert()
        for group in ("layerwise_blocks", "refiner_blocks"):
            self.assertIn(f"text_fusion.{group}.0.attn.to_q.weight", result)
            self.assertIn(f"text_fusion.{group}.0.attn.to_out.0.weight", result)
            self.assertIn(f"text_fusion.{group}.0.attn.to_gate.weight", result)

    def test_text_fusion_ff_keys(self):
        result = self._convert()
        for group in ("layerwise_blocks", "refiner_blocks"):
            self.assertIn(f"text_fusion.{group}.0.ff.gate.weight", result)
            self.assertIn(f"text_fusion.{group}.0.ff.up.weight", result)
            self.assertIn(f"text_fusion.{group}.0.ff.down.weight", result)

    def test_text_fusion_projector(self):
        result = self._convert()
        self.assertIn("text_fusion.projector.weight", result)
        self.assertNotIn("txtfusion.projector.weight", result)

    def test_original_txtfusion_keys_absent(self):
        result = self._convert()
        self.assertFalse(any("txtfusion." in k for k in result))

    # ------------------------------------------------------------------
    # Pass-through keys (same name in both formats)
    # ------------------------------------------------------------------

    def test_img_in_unchanged(self):
        result = self._convert()
        self.assertIn("img_in.weight", result)
        self.assertIn("img_in.bias", result)

    def test_time_embed_unchanged(self):
        result = self._convert()
        self.assertIn("time_embed.linear_1.weight", result)
        self.assertIn("time_embed.linear_2.weight", result)

    def test_time_mod_proj_unchanged(self):
        result = self._convert()
        self.assertIn("time_mod_proj.weight", result)

    def test_txt_in_unchanged(self):
        result = self._convert()
        self.assertIn("txt_in.norm.weight", result)
        self.assertIn("txt_in.linear_1.weight", result)

    def test_final_layer_unchanged(self):
        result = self._convert()
        self.assertIn("final_layer.norm.weight", result)
        self.assertIn("final_layer.linear.weight", result)
        self.assertIn("final_layer.scale_shift_table", result)

    def test_rotary_emb_has_no_state_dict_entries(self):
        # Krea2RotaryPosEmbed computes frequencies on the fly with no parameters or
        # persistent buffers, so it contributes nothing to the state dict.
        result = self._convert()
        self.assertFalse(any(k.startswith("rotary_emb.") for k in result))

    # ------------------------------------------------------------------
    # Value identity: conversion must not alter tensor data
    # ------------------------------------------------------------------

    def test_passthrough_values_are_identical(self):
        orig_weight = torch.randn(32, 16)
        sd = {"diffusion_model.img_in.weight": orig_weight}
        result = convert_krea2_transformer_checkpoint_to_diffusers(sd)
        self.assertTrue(torch.equal(result["img_in.weight"], orig_weight))

    def test_remapped_values_are_identical(self):
        orig_weight = torch.randn(32, 32)
        sd = {"diffusion_model.blocks.0.attn.wq.weight": orig_weight}
        result = convert_krea2_transformer_checkpoint_to_diffusers(sd)
        self.assertTrue(torch.equal(result["transformer_blocks.0.attn.to_q.weight"], orig_weight))

    # ------------------------------------------------------------------
    # Round-trip key coverage: every diffusers state dict key is produced
    # ------------------------------------------------------------------

    def test_output_keys_match_diffusers_model_state_dict(self):
        """
        Instantiate a tiny Krea2 model, get its diffusers-format state dict,
        and verify the conversion function produces the same key set when given
        a synthetic original-format checkpoint with matching shapes.
        """
        init_cfg = {
            "in_channels": 16,
            "num_layers": 2,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "timestep_embed_dim": 8,
            "text_hidden_dim": 16,
            "num_text_layers": 3,
            "text_num_attention_heads": 2,
            "text_num_key_value_heads": 1,
            "text_intermediate_size": 16,
            "num_layerwise_text_blocks": 1,
            "num_refiner_text_blocks": 1,
            "axes_dims_rope": (4, 2, 2),
            "rope_theta": 1000.0,
            "norm_eps": 1e-5,
        }
        model = Krea2Transformer2DModel(**init_cfg)
        diffusers_keys = set(model.state_dict().keys())

        original_sd = _make_krea2_original_state_dict(
            num_layers=2,
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            timestep_embed_dim=8,
            text_hidden_dim=16,
            num_text_layers=3,
            text_num_attention_heads=2,
            text_num_key_value_heads=1,
            text_intermediate_size=16,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            in_channels=16,
            prefix="diffusion_model.",
        )
        converted = convert_krea2_transformer_checkpoint_to_diffusers(original_sd)
        converted_keys = set(converted.keys())

        missing = diffusers_keys - converted_keys
        extra = converted_keys - diffusers_keys
        self.assertEqual(missing, set(), msg=f"Keys in diffusers model but missing from conversion: {missing}")
        self.assertEqual(extra, set(), msg=f"Keys produced by conversion but not in diffusers model: {extra}")


# ---------------------------------------------------------------------------
# Integration test — requires actual single-file checkpoint (nightly / big GPU)
# ---------------------------------------------------------------------------

# TODO: fill in once a public Krea2 single-file checkpoint is released.
# class TestKrea2TransformerSingleFile(SingleFileModelTesterMixin):
#     model_class = Krea2Transformer2DModel
#     ckpt_path = "https://huggingface.co/<org>/<repo>/blob/main/krea2_transformer.safetensors"
#     repo_id = "<org>/<diffusers-format-repo>"
#     subfolder = "transformer"


if __name__ == "__main__":
    unittest.main()
