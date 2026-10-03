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

The generic model test mixins (`ModelTesterMixin`, `TrainingTesterMixin`, `MemoryTesterMixin`,
`TorchCompileTesterMixin`) cover save/load, determinism, dtype casting, gradient checkpointing, and
compilation via `get_init_dict()` / `get_dummy_inputs()`. `AttentionTesterMixin` is intentionally not used:
its generic `test_fuse_unfuse_qkv_projections` assumes any `Attention` submodule exposing `to_qkv`/`to_kv`
supports fusion, which doesn't hold here — the SA3 attention modules set `_supports_qkv_fusion = False`
because their fused projections are differential-attention groups (`[q1|q2|k1|k2|v]`), not the plain
`to_q`/`to_k`/`to_v` layout `fuse_projections` assumes.

This file also covers what those mixins don't reach: structural parity with the released SA3 Medium
checkpoint, the local-additive (inpaint) conditioning path, and encoder-attention-mask / timestep-boundary
behavior.
"""

import unittest

import torch
from accelerate import init_empty_weights

from diffusers import StableAudio3DiTModel

from ...testing_utils import require_torch, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


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
BATCH_SIZE = 2
T_AUDIO = 16
T_TEXT = 4
GLOBAL_DIM = TINY_CFG["global_cond_dim"]
COND_DIM = TINY_CFG["cond_token_dim"]
IO_CHANNELS = TINY_CFG["io_channels"]


def _make_inputs(batch: int = BATCH_SIZE, device: str = "cpu"):
    torch.manual_seed(0)
    hidden_states = torch.randn(batch, IO_CHANNELS, T_AUDIO, device=device)
    timestep = torch.rand(batch, device=device)
    encoder_hidden_states = torch.randn(batch, T_TEXT, COND_DIM, device=device)
    global_hidden_states = torch.randn(batch, GLOBAL_DIM, device=device)
    return hidden_states, timestep, encoder_hidden_states, global_hidden_states


class StableAudio3DiTTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return StableAudio3DiTModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (IO_CHANNELS, T_AUDIO)

    def get_init_dict(self) -> dict:
        return dict(TINY_CFG)

    def get_dummy_inputs(self) -> dict:
        hidden_states, timestep, encoder_hidden_states, global_hidden_states = _make_inputs(
            batch=BATCH_SIZE, device=torch_device
        )
        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "global_hidden_states": global_hidden_states,
        }


class TestStableAudio3DiTModel(StableAudio3DiTTesterConfig, ModelTesterMixin):
    pass


class TestStableAudio3DiTModelTraining(StableAudio3DiTTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        # `gradient_checkpointing` lives on the top-level model (it wraps the block loop in `forward`), not
        # on `StableAudio3DiTBlock` itself.
        super().test_gradient_checkpointing_is_applied(expected_set={"StableAudio3DiTModel"})


class TestStableAudio3DiTModelMemory(StableAudio3DiTTesterConfig, MemoryTesterMixin):
    pass


class TestStableAudio3DiTModelCompile(StableAudio3DiTTesterConfig, TorchCompileTesterMixin):
    pass


@require_torch
class TestStableAudio3DiTModelBehavior(unittest.TestCase):
    """Coverage beyond the generic mixins: encoder-attention-mask handling, timestep-boundary values,
    `patch_size` / `use_differential_attention` variants, batch-size handling, the local-additive
    (inpaint) conditioning path, and memory-token registration."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = StableAudio3DiTModel(**TINY_CFG).eval()

    def test_attention_mask(self):
        hs, t, ctx, glob = _make_inputs()
        mask = torch.ones(BATCH_SIZE, T_TEXT, dtype=torch.bool)
        mask[0, -1] = False  # mask out last token for first sample
        out = self.model(hs, t, ctx, glob, encoder_attention_mask=mask)
        self.assertEqual(out.sample.shape, hs.shape)

    def test_different_batch_sizes(self):
        for batch in (1, 3, 4):
            hs, t, ctx, glob = _make_inputs(batch=batch)
            out = self.model(hs, t, ctx, glob)
            self.assertEqual(out.sample.shape[0], batch)

    def test_timestep_boundary_zero(self):
        """t=0 should not produce NaN (logSNR clamp handles edge)."""
        hs, _, ctx, glob = _make_inputs()
        t = torch.zeros(BATCH_SIZE)
        out = self.model(hs, t, ctx, glob)
        self.assertFalse(out.sample.isnan().any())

    def test_timestep_boundary_one(self):
        """t=1 should not produce NaN."""
        hs, _, ctx, glob = _make_inputs()
        t = torch.ones(BATCH_SIZE)
        out = self.model(hs, t, ctx, glob)
        self.assertFalse(out.sample.isnan().any())

    def test_gradient_checkpointing_toggle(self):
        """enable_gradient_checkpointing / disable should not raise, independent of accelerator availability."""
        self.model.enable_gradient_checkpointing()
        self.assertTrue(self.model.gradient_checkpointing)
        self.model.disable_gradient_checkpointing()
        self.assertFalse(self.model.gradient_checkpointing)

    def test_patch_size_2(self):
        """patch_size=2 should halve T in the transformer and restore it on output."""
        model = StableAudio3DiTModel(**dict(TINY_CFG, patch_size=2))
        hs, t, ctx, glob = _make_inputs()
        out = model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape, hs.shape)

    def test_differential_attention(self):
        """use_differential_attention=True should produce valid output shape."""
        model = StableAudio3DiTModel(**dict(TINY_CFG, use_differential_attention=True))
        hs, t, ctx, glob = _make_inputs()
        out = model(hs, t, ctx, glob)
        self.assertEqual(out.sample.shape, hs.shape)

    def test_local_add_cond_zero_init_is_noop(self):
        """`to_local_embed` is zero-initialised, so inpaint conditioning is a no-op until trained."""
        hs, t, ctx, glob = _make_inputs()
        local = torch.randn(BATCH_SIZE, LOCAL_ADD_COND_DIM, T_AUDIO)
        out_base = self.model(hs, t, ctx, glob).sample
        out_local = self.model(hs, t, ctx, glob, local_add_cond=local).sample
        self.assertEqual(out_local.shape, hs.shape)
        self.assertTrue(torch.allclose(out_base, out_local))

    def test_local_add_cond_changes_output_when_trained(self):
        """With non-zero `to_local_embed` weights, inpaint conditioning must alter the output."""
        with torch.no_grad():
            for block in self.model.transformer_blocks:
                block.to_local_embed[-1].weight.normal_()
                block.to_local_embed[-1].bias.normal_()
        hs, t, ctx, glob = _make_inputs()
        local = torch.randn(BATCH_SIZE, LOCAL_ADD_COND_DIM, T_AUDIO)
        out_base = self.model(hs, t, ctx, glob).sample
        out_local = self.model(hs, t, ctx, glob, local_add_cond=local).sample
        self.assertFalse(torch.allclose(out_base, out_local))

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
        # Learned text-padding embedding (checkpoint key: conditioner.conditioners.prompt.padding_embedding).
        "prompt_padding_embedding": (768,),
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
        with init_empty_weights():
            model = StableAudio3DiTModel(**PROD_CFG)
        actual = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        expected = _expected_prod_state_dict_shapes()

        missing = set(expected) - set(actual)
        unexpected = set(actual) - set(expected)
        self.assertEqual(missing, set(), f"checkpoint tensors missing from model: {sorted(missing)[:10]}")
        self.assertEqual(unexpected, set(), f"model has tensors not in checkpoint: {sorted(unexpected)[:10]}")

        mismatched = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
        self.assertEqual(mismatched, {}, f"shape mismatches: {list(mismatched.items())[:10]}")

        # The released SA3 Medium DiT has 522 tensors, plus the learned text-padding embedding
        # (relocated from the reference conditioner into the diffusers DiT) → 523.
        self.assertEqual(len(actual), 523)


if __name__ == "__main__":
    unittest.main()
