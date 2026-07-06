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
Tests for Ideogram4Transformer2DModel from_single_file() support.

Two suites:
  1. TestIdeogram4ConversionFunction  — offline unit tests for the key-mapping
     function. No checkpoint, no GPU, no network.
  2. TestIdeogram4TransformerSingleFile — @nightly integration test that loads
     an actual single-file checkpoint and compares against from_pretrained().
     Requires a public checkpoint; fill in ckpt_path / repo_id when available.
"""

import unittest

import torch

from diffusers import Ideogram4Transformer2DModel
from diffusers.loaders.single_file_utils import convert_ideogram4_transformer_checkpoint_to_diffusers

from ..testing_utils import enable_full_determinism


enable_full_determinism()


# ---------------------------------------------------------------------------
# Helper: build a synthetic "original" state dict for a small Ideogram4 model
# ---------------------------------------------------------------------------


def _make_ideogram4_original_state_dict(
    num_layers=2,
    hidden_size=32,
    num_heads=4,
    intermediate_size=32,
    adaln_dim=16,
    in_channels=16,
    llm_features_dim=24,
    prefix="diffusion_model.",
):
    """
    Produces a synthetic state dict in the pre-diffusers Ideogram4 format:
      - fused attention.qkv   (3 * head_dim * num_heads, hidden_size)
      - attention.o           (hidden_size, hidden_size)
      - feed_forward.w1/w2/w3
      - adaln_modulation
      - per-layer norms
      - top-level embeddings / final_layer
    All tensors contain random data; shapes match what the model actually expects.
    """
    head_dim = hidden_size // num_heads
    sd = {}

    # Top-level modules
    sd[f"{prefix}input_proj.weight"] = torch.randn(hidden_size, in_channels)
    sd[f"{prefix}input_proj.bias"] = torch.randn(hidden_size)
    sd[f"{prefix}llm_cond_norm.weight"] = torch.randn(llm_features_dim)
    sd[f"{prefix}llm_cond_proj.weight"] = torch.randn(hidden_size, llm_features_dim)
    sd[f"{prefix}llm_cond_proj.bias"] = torch.randn(hidden_size)
    # t_embedding is Ideogram4EmbedScalar: two linear layers (mlp_in, mlp_out) both with bias
    sd[f"{prefix}t_embedding.mlp_in.weight"] = torch.randn(hidden_size, hidden_size)
    sd[f"{prefix}t_embedding.mlp_in.bias"] = torch.randn(hidden_size)
    sd[f"{prefix}t_embedding.mlp_out.weight"] = torch.randn(hidden_size, hidden_size)
    sd[f"{prefix}t_embedding.mlp_out.bias"] = torch.randn(hidden_size)
    sd[f"{prefix}adaln_proj.weight"] = torch.randn(adaln_dim, hidden_size)
    sd[f"{prefix}adaln_proj.bias"] = torch.randn(adaln_dim)
    # embed_image_indicator is nn.Embedding (weight only, no bias)
    sd[f"{prefix}embed_image_indicator.weight"] = torch.randn(2, hidden_size)
    # rotary_emb uses register_buffer with persistent=False -> not in state_dict

    for i in range(num_layers):
        p = f"{prefix}layers.{i}."
        # Fused QKV (original format, no bias)
        sd[f"{p}attention.qkv.weight"] = torch.randn(3 * hidden_size, hidden_size)
        # Output projection (original name: attention.o, no bias)
        sd[f"{p}attention.o.weight"] = torch.randn(hidden_size, hidden_size)
        # q/k norms (RMSNorm with elementwise_affine=True, same names in both formats)
        sd[f"{p}attention.norm_q.weight"] = torch.randn(head_dim)
        sd[f"{p}attention.norm_k.weight"] = torch.randn(head_dim)
        # Feed-forward (no bias)
        sd[f"{p}feed_forward.w1.weight"] = torch.randn(intermediate_size, hidden_size)
        sd[f"{p}feed_forward.w2.weight"] = torch.randn(hidden_size, intermediate_size)
        sd[f"{p}feed_forward.w3.weight"] = torch.randn(intermediate_size, hidden_size)
        # AdaLN modulation (has bias)
        sd[f"{p}adaln_modulation.weight"] = torch.randn(4 * hidden_size, adaln_dim)
        sd[f"{p}adaln_modulation.bias"] = torch.randn(4 * hidden_size)
        # RMSNorms (same names in both formats)
        for norm in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
            sd[f"{p}{norm}.weight"] = torch.randn(hidden_size)

    # Final layer (same names in both formats)
    # norm_final uses elementwise_affine=False -> no weight/bias in state_dict
    sd[f"{prefix}final_layer.linear.weight"] = torch.randn(in_channels, hidden_size)
    sd[f"{prefix}final_layer.linear.bias"] = torch.randn(in_channels)
    # adaln_modulation: nn.Linear(adaln_dim, hidden_size, bias=True)
    sd[f"{prefix}final_layer.adaln_modulation.weight"] = torch.randn(hidden_size, adaln_dim)
    sd[f"{prefix}final_layer.adaln_modulation.bias"] = torch.randn(hidden_size)

    return sd


class TestIdeogram4ConversionFunction(unittest.TestCase):
    """Offline unit tests for convert_ideogram4_transformer_checkpoint_to_diffusers."""

    def _convert(self, prefix="diffusion_model.", num_layers=2):
        sd = _make_ideogram4_original_state_dict(prefix=prefix, num_layers=num_layers)
        return convert_ideogram4_transformer_checkpoint_to_diffusers(sd)

    # ------------------------------------------------------------------
    # Prefix stripping
    # ------------------------------------------------------------------

    def test_strips_diffusion_model_prefix(self):
        result = self._convert(prefix="diffusion_model.")
        self.assertFalse(any(k.startswith("diffusion_model.") for k in result))

    def test_strips_conditional_transformer_prefix(self):
        sd = _make_ideogram4_original_state_dict(prefix="conditional_transformer.")
        result = convert_ideogram4_transformer_checkpoint_to_diffusers(sd)
        self.assertFalse(any(k.startswith("conditional_transformer.") for k in result))

    def test_no_prefix_passthrough(self):
        # If there is no known prefix the keys should survive untouched.
        sd = _make_ideogram4_original_state_dict(prefix="")
        result = convert_ideogram4_transformer_checkpoint_to_diffusers(sd)
        self.assertIn("layers.0.attention.to_q.weight", result)

    # ------------------------------------------------------------------
    # QKV split
    # ------------------------------------------------------------------

    def test_fused_qkv_is_split_into_three_projections(self):
        result = self._convert()
        for i in range(2):
            self.assertIn(f"layers.{i}.attention.to_q.weight", result)
            self.assertIn(f"layers.{i}.attention.to_k.weight", result)
            self.assertIn(f"layers.{i}.attention.to_v.weight", result)
            self.assertNotIn(f"layers.{i}.attention.qkv.weight", result)

    def test_split_shapes_are_equal_thirds(self):
        result = self._convert()
        # Each split should be hidden_size // 3 of the original fused dim (hidden_size=32 -> 32 each)
        for proj in ("to_q", "to_k", "to_v"):
            self.assertEqual(result[f"layers.0.attention.{proj}.weight"].shape, (32, 32))

    def test_split_values_are_contiguous_chunks(self):
        # Build a deterministic fused weight and verify the three slices.
        fused = torch.arange(96, dtype=torch.float32).reshape(96, 1)
        sd = {"diffusion_model.layers.0.attention.qkv.weight": fused}
        result = convert_ideogram4_transformer_checkpoint_to_diffusers(sd)
        expected_q = fused[:32]
        expected_k = fused[32:64]
        expected_v = fused[64:]
        self.assertTrue(torch.equal(result["layers.0.attention.to_q.weight"], expected_q))
        self.assertTrue(torch.equal(result["layers.0.attention.to_k.weight"], expected_k))
        self.assertTrue(torch.equal(result["layers.0.attention.to_v.weight"], expected_v))

    # ------------------------------------------------------------------
    # Output projection rename: attention.o -> attention.to_out.0
    # ------------------------------------------------------------------

    def test_attention_o_renamed_to_to_out_0(self):
        result = self._convert()
        for i in range(2):
            self.assertIn(f"layers.{i}.attention.to_out.0.weight", result)
            self.assertNotIn(f"layers.{i}.attention.o.weight", result)

    # ------------------------------------------------------------------
    # Keys that should pass through unchanged
    # ------------------------------------------------------------------

    def test_feed_forward_keys_unchanged(self):
        result = self._convert()
        for i in range(2):
            for w in ("w1", "w2", "w3"):
                self.assertIn(f"layers.{i}.feed_forward.{w}.weight", result)

    def test_norm_keys_unchanged(self):
        result = self._convert()
        for i in range(2):
            for norm in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
                self.assertIn(f"layers.{i}.{norm}.weight", result)

    def test_adaln_modulation_unchanged(self):
        result = self._convert()
        for i in range(2):
            self.assertIn(f"layers.{i}.adaln_modulation.weight", result)

    def test_final_layer_keys_unchanged(self):
        result = self._convert()
        # norm_final uses elementwise_affine=False, so it has no parameters
        self.assertNotIn("final_layer.norm_final.weight", result)
        self.assertIn("final_layer.linear.weight", result)
        self.assertIn("final_layer.linear.bias", result)
        self.assertIn("final_layer.adaln_modulation.weight", result)
        self.assertIn("final_layer.adaln_modulation.bias", result)

    def test_top_level_embedding_keys_unchanged(self):
        result = self._convert()
        for key in (
            "input_proj.weight",
            "input_proj.bias",
            "llm_cond_norm.weight",
            "llm_cond_proj.weight",
            "llm_cond_proj.bias",
            # t_embedding is Ideogram4EmbedScalar: two linear layers
            "t_embedding.mlp_in.weight",
            "t_embedding.mlp_in.bias",
            "t_embedding.mlp_out.weight",
            "t_embedding.mlp_out.bias",
            "adaln_proj.weight",
            "adaln_proj.bias",
            "embed_image_indicator.weight",
        ):
            self.assertIn(key, result)

    # ------------------------------------------------------------------
    # Round-trip key coverage: every diffusers state dict key is produced
    # ------------------------------------------------------------------

    def test_output_keys_match_diffusers_model_state_dict(self):
        """
        Instantiate a tiny Ideogram4 model, read its diffusers-format state dict,
        then verify that the conversion function produces exactly the same key set
        when given a synthetic original-format checkpoint.
        """
        init_cfg = {
            "in_channels": 16,
            "num_layers": 2,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "adaln_dim": 16,
            "llm_features_dim": 24,
            "rope_theta": 10_000,
            "mrope_section": (2, 1, 1),
            "norm_eps": 1e-5,
        }
        model = Ideogram4Transformer2DModel(**init_cfg)
        diffusers_keys = set(model.state_dict().keys())

        original_sd = _make_ideogram4_original_state_dict(
            num_layers=2,
            hidden_size=32,  # num_heads * head_dim = 4 * 8
            num_heads=4,
            intermediate_size=32,
            adaln_dim=16,
            in_channels=16,
            llm_features_dim=24,
            prefix="diffusion_model.",
        )
        converted = convert_ideogram4_transformer_checkpoint_to_diffusers(original_sd)
        converted_keys = set(converted.keys())

        missing = diffusers_keys - converted_keys
        extra = converted_keys - diffusers_keys
        self.assertEqual(missing, set(), msg=f"Keys in diffusers model but missing from conversion: {missing}")
        self.assertEqual(extra, set(), msg=f"Keys produced by conversion but not in diffusers model: {extra}")


# ---------------------------------------------------------------------------
# Integration test — requires actual single-file checkpoint (nightly / big GPU)
# ---------------------------------------------------------------------------

# TODO: fill in once a public Ideogram4 single-file checkpoint is released.
# class TestIdeogram4TransformerSingleFile(SingleFileModelTesterMixin):
#     model_class = Ideogram4Transformer2DModel
#     ckpt_path = "https://huggingface.co/<org>/<repo>/blob/main/ideogram4_transformer.safetensors"
#     repo_id = "<org>/<diffusers-format-repo>"
#     subfolder = "transformer"


if __name__ == "__main__":
    unittest.main()
