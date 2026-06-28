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

"""
Unit tests for StableAudio3DiTModel.

Covers:
  - Output shape matches input shape
  - return_dict=False returns a plain tuple
  - Gradient checkpointing can be toggled without errors
  - Different batch sizes work
  - Timestep boundary values (t=0 and t=1)
  - Attention mask is accepted without error
  - Config round-trip: save / reload produces identical output
  - Device placement
"""

import unittest

import torch

from diffusers import StableAudio3DiTModel

from ...testing_utils import require_torch, torch_device


# ──────────────────────────────────────────────────────────────────────────────
# Tiny config — fast, CPU-runnable
# ──────────────────────────────────────────────────────────────────────────────

TINY_CFG = {
    "io_channels": 8,
    "patch_size": 1,
    "embed_dim": 32,
    "depth": 2,
    "num_heads": 4,  # dim_heads = 32 // 4 = 8
    "cond_token_dim": 16,
    "global_cond_dim": 12,
    "local_add_cond_dim": 5,
    "timestep_features_dim": 16,
    "ff_mult": 2,
    "num_memory_tokens": 3,
    "use_differential_attention": False,
}

LOCAL_ADD_COND_DIM = TINY_CFG["local_add_cond_dim"]

# For the tiny config: io_channels=8, T_audio=16, T_text=4
T_AUDIO = 16
T_TEXT = 4
GLOBAL_DIM = TINY_CFG["global_cond_dim"]
COND_DIM = TINY_CFG["cond_token_dim"]
IO_CHANNELS = TINY_CFG["io_channels"]


def _make_model(cfg: dict = None, device: str = "cpu") -> StableAudio3DiTModel:
    if cfg is None:
        cfg = TINY_CFG
    model = StableAudio3DiTModel(**cfg).to(device).eval()
    return model


def _make_inputs(batch: int = 2, device: str = "cpu"):
    torch.manual_seed(0)
    hidden_states = torch.randn(batch, IO_CHANNELS, T_AUDIO, device=device)
    timestep = torch.rand(batch, device=device)
    encoder_hidden_states = torch.randn(batch, T_TEXT, COND_DIM, device=device)
    global_hidden_states = torch.randn(batch, GLOBAL_DIM, device=device)
    return hidden_states, timestep, encoder_hidden_states, global_hidden_states


@require_torch
class TestStableAudio3DiTModelTinyConfig(unittest.TestCase):
    """Fast unit tests on a tiny model configuration."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = _make_model()
        self.batch = 2

    # ------------------------------------------------------------------

    def test_output_shape(self):
        hs, t, ctx, glob = _make_inputs(self.batch)
        out = self.model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape, hs.shape)

    # ------------------------------------------------------------------

    def test_return_dict_false(self):
        hs, t, ctx, glob = _make_inputs(self.batch)
        out = self.model(hs, t, ctx, glob, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, hs.shape)

    # ------------------------------------------------------------------

    def test_attention_mask(self):
        hs, t, ctx, glob = _make_inputs(self.batch)
        mask = torch.ones(self.batch, T_TEXT, dtype=torch.bool)
        mask[0, -1] = False  # mask out last token for first sample
        out = self.model(hs, t, ctx, glob, encoder_attention_mask=mask)
        self.assertEqual(out.sample.shape, hs.shape)

    # ------------------------------------------------------------------

    def test_batch_size_1(self):
        hs, t, ctx, glob = _make_inputs(1)
        out = self.model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape[0], 1)

    # ------------------------------------------------------------------

    def test_batch_size_4(self):
        hs, t, ctx, glob = _make_inputs(4)
        out = self.model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape[0], 4)

    # ------------------------------------------------------------------

    def test_timestep_boundary_zero(self):
        """t=0 should not produce NaN (logSNR clamp handles edge)."""
        hs, _, ctx, glob = _make_inputs(self.batch)
        t = torch.zeros(self.batch)
        out = self.model(hs, t, ctx, glob)
        self.assertFalse(out.sample.isnan().any())

    # ------------------------------------------------------------------

    def test_timestep_boundary_one(self):
        """t=1 should not produce NaN."""
        hs, _, ctx, glob = _make_inputs(self.batch)
        t = torch.ones(self.batch)
        out = self.model(hs, t, ctx, glob)
        self.assertFalse(out.sample.isnan().any())

    # ------------------------------------------------------------------

    def test_gradient_checkpointing_toggle(self):
        """enable_gradient_checkpointing / disable should not raise."""
        self.model.enable_gradient_checkpointing()
        self.assertTrue(self.model.gradient_checkpointing)
        self.model.disable_gradient_checkpointing()
        self.assertFalse(self.model.gradient_checkpointing)

    # ------------------------------------------------------------------

    def test_config_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            reloaded = StableAudio3DiTModel.from_pretrained(tmpdir).eval()

        hs, t, ctx, glob = _make_inputs(self.batch)
        with torch.no_grad():
            out_orig = self.model(hs, t, ctx, glob).sample
            out_rel = reloaded(hs, t, ctx, glob).sample

        self.assertTrue(torch.allclose(out_orig, out_rel, atol=1e-5))

    # ------------------------------------------------------------------

    def test_patch_size_2(self):
        """patch_size=2 should halve T in the transformer and restore it on output."""
        cfg = dict(TINY_CFG, patch_size=2)
        model = _make_model(cfg)
        hs, t, ctx, glob = _make_inputs(self.batch)
        out = model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape, hs.shape)

    # ------------------------------------------------------------------

    def test_differential_attention(self):
        """use_differential_attention=True should produce valid output shape."""
        cfg = dict(TINY_CFG, use_differential_attention=True)
        model = _make_model(cfg)
        hs, t, ctx, glob = _make_inputs(self.batch)
        out = model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape, hs.shape)

    # ------------------------------------------------------------------

    def test_no_nan_in_output(self):
        """Smoke test: deterministic inputs produce no NaN."""
        hs, t, ctx, glob = _make_inputs(self.batch)
        out = self.model(hs, t, ctx, glob)
        self.assertFalse(out.sample.isnan().any())
        self.assertFalse(out.sample.isinf().any())

    # ------------------------------------------------------------------

    def test_local_add_cond_zero_init_is_noop(self):
        """`to_local_embed` is zero-initialised, so inpaint conditioning is a no-op until trained."""
        hs, t, ctx, glob = _make_inputs(self.batch)
        local = torch.randn(self.batch, LOCAL_ADD_COND_DIM, T_AUDIO)
        out_base = self.model(hs, t, ctx, glob).sample
        out_local = self.model(hs, t, ctx, glob, local_add_cond=local).sample
        self.assertEqual(out_local.shape, hs.shape)
        self.assertTrue(torch.allclose(out_base, out_local))

    # ------------------------------------------------------------------

    def test_local_add_cond_changes_output_when_trained(self):
        """With non-zero `to_local_embed` weights, inpaint conditioning must alter the output."""
        with torch.no_grad():
            for block in self.model.transformer_blocks:
                block.to_local_embed[-1].weight.normal_()
                block.to_local_embed[-1].bias.normal_()
        hs, t, ctx, glob = _make_inputs(self.batch)
        local = torch.randn(self.batch, LOCAL_ADD_COND_DIM, T_AUDIO)
        out_base = self.model(hs, t, ctx, glob).sample
        out_local = self.model(hs, t, ctx, glob, local_add_cond=local).sample
        self.assertFalse(torch.allclose(out_base, out_local))

    # ------------------------------------------------------------------

    def test_memory_tokens_present(self):
        """Memory tokens parameter exists with the configured count."""
        self.assertEqual(self.model.memory_tokens.shape[0], TINY_CFG["num_memory_tokens"])
        self.assertEqual(self.model.memory_tokens.shape[1], TINY_CFG["embed_dim"])


# ──────────────────────────────────────────────────────────────────────────────
# Structural parity with the released SA3 Medium checkpoint
# ──────────────────────────────────────────────────────────────────────────────

# Production config from stabilityai/stable-audio-3-medium model_config.json.
PROD_CFG = {
    "io_channels": 256,
    "patch_size": 1,
    "embed_dim": 1536,
    "depth": 24,
    "num_heads": 24,
    "cond_token_dim": 768,
    "global_cond_dim": 768,
    "local_add_cond_dim": 257,
    "timestep_features_dim": 256,
    "ff_mult": 4.0,
    "num_memory_tokens": 64,
    "use_differential_attention": True,
}


def _expected_prod_state_dict_shapes() -> dict:
    """The exact parameter/buffer names and shapes of the SA3 Medium DiT checkpoint,
    expressed in diffusers naming (reference keys renamed: ``ff.ff.0.proj`` → ``ff.proj_in``,
    ``ff.ff.2`` → ``ff.proj_out``; ``model.model.transformer.layers.{i}`` → ``transformer_blocks.{i}``).

    Derived from the real checkpoint tensor index (997 tensors total; 522 in the DiT).
    """
    E, H = 1536, 64  # embed_dim, dim_heads
    top = {
        "to_timestep_embed.0.weight": (E, 256),
        "to_timestep_embed.0.bias": (E,),
        "to_timestep_embed.2.weight": (E, E),
        "to_timestep_embed.2.bias": (E,),
        "to_cond_embed.0.weight": (E, 768),
        "to_cond_embed.2.weight": (E, E),
        "to_global_embed.0.weight": (E, 768),
        "to_global_embed.2.weight": (E, E),
        "global_cond_embedder.0.weight": (E, E),
        "global_cond_embedder.0.bias": (E,),
        "global_cond_embedder.2.weight": (6 * E, E),
        "global_cond_embedder.2.bias": (6 * E,),
        "preprocess_conv.weight": (256, 256, 1),
        "postprocess_conv.weight": (256, 256, 1),
        "proj_in.weight": (E, 256),
        "proj_out.weight": (256, E),
        "memory_tokens": (64, E),
        "rotary_pos_emb.inv_freq": (H // 4,),
    }
    per_block = {
        "pre_norm.gamma": (E,),
        "self_attn.to_qkv.weight": (5 * E, E),
        "self_attn.to_out.weight": (E, E),
        "self_attn.q_norm.gamma": (H,),
        "self_attn.k_norm.gamma": (H,),
        "cross_attend_norm.gamma": (E,),
        "cross_attn.to_q.weight": (2 * E, E),
        "cross_attn.to_kv.weight": (3 * E, E),
        "cross_attn.to_out.weight": (E, E),
        "cross_attn.q_norm.gamma": (H,),
        "cross_attn.k_norm.gamma": (H,),
        "ff_norm.gamma": (E,),
        "ff.proj_in.weight": (2 * 4 * E, E),
        "ff.proj_in.bias": (2 * 4 * E,),
        "ff.proj_out.weight": (E, 4 * E),
        "ff.proj_out.bias": (E,),
        "to_local_embed.0.weight": (E, 257),
        "to_local_embed.0.bias": (E,),
        "to_local_embed.2.weight": (E, E),
        "to_local_embed.2.bias": (E,),
        "to_scale_shift_gate": (6 * E,),
    }
    expected = dict(top)
    for i in range(24):
        for name, shape in per_block.items():
            expected[f"transformer_blocks.{i}.{name}"] = shape
    return expected


@require_torch
class TestStableAudio3DiTProductionStructure(unittest.TestCase):
    """The production-config model must match the released checkpoint's tensor names and shapes
    exactly, so converted SA3 Medium weights load with ``strict=True``."""

    def test_state_dict_matches_checkpoint(self):
        model = StableAudio3DiTModel(**PROD_CFG)
        actual = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        expected = _expected_prod_state_dict_shapes()

        missing = set(expected) - set(actual)
        unexpected = set(actual) - set(expected)
        self.assertEqual(missing, set(), f"checkpoint tensors missing from model: {sorted(missing)[:10]}")
        self.assertEqual(unexpected, set(), f"model has tensors not in checkpoint: {sorted(unexpected)[:10]}")

        mismatched = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
        self.assertEqual(mismatched, {}, f"shape mismatches: {list(mismatched.items())[:10]}")

        # The released SA3 Medium DiT has 522 tensors.
        self.assertEqual(len(actual), 522)


@require_torch
class TestStableAudio3DiTModelOnDevice(unittest.TestCase):
    """Test model runs on whichever device pytest selects."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch_device
        self.model = _make_model(device=str(self.device))

    def test_output_on_device(self):
        hs, t, ctx, glob = _make_inputs(1, device=str(self.device))
        out = self.model(hs, t, ctx, glob)
        self.assertEqual(out.sample.device.type, torch.device(self.device).type)

    def test_output_shape_on_device(self):
        hs, t, ctx, glob = _make_inputs(2, device=str(self.device))
        out = self.model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape, hs.shape)


if __name__ == "__main__":
    unittest.main()
