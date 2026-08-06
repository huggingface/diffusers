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

import logging
import unittest

import torch

from diffusers.hooks import RollingKVCacheConfig, apply_rolling_kv_cache, get_rolling_kv_cache_state
from diffusers.hooks.rolling_kv_cache import (
    _ROLLING_KV_CACHE_HOOK,
    RollingKVAttentionProcessor,
    RollingKVCacheBlockState,
    RollingKVCacheState,
)
from diffusers.models.cache_utils import CacheMixin


_DEVICE = torch.device("cpu")
_HEAD_DIM = 4


# ---------- Fake self-attention so cache contents read out as input tokens ----------


class _IdentitySelfAttention(torch.nn.Module):
    """Self-attention stub with identity Q/K/V/norm/out projections.

    Because every projection is identity, whatever scalar token value goes in comes back out
    inside `cached_key` / `cached_value`. That makes the cache directly inspectable in tests.
    """

    def __init__(self):
        super().__init__()
        self.heads = 1
        self.is_cross_attention = False
        self.to_q = torch.nn.Identity()
        self.to_k = torch.nn.Identity()
        self.to_v = torch.nn.Identity()
        self.norm_q = torch.nn.Identity()
        self.norm_k = torch.nn.Identity()
        self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])


class _FakeTransformer(torch.nn.Module, CacheMixin):
    """Tiny CacheMixin wrapper so `cache_context(...)` works in the tests."""

    def __init__(self):
        super().__init__()
        self.attn = _IdentitySelfAttention()

    def forward(self, hidden_states):
        return self.attn(hidden_states)


def _make_transformer(window_size: int = -1) -> _FakeTransformer:
    transformer = _FakeTransformer().to(_DEVICE).eval()
    # Silence the "untested attention class" warning the hook emits for non-Wan modules —
    # exercising that warning is covered by `test_warns_when_attaching_to_untested_class`.
    logging.getLogger("diffusers.hooks.rolling_kv_cache").setLevel(logging.ERROR)
    apply_rolling_kv_cache(transformer, RollingKVCacheConfig(window_size=window_size))
    logging.getLogger("diffusers.hooks.rolling_kv_cache").setLevel(logging.WARNING)
    return transformer


def _ramp(values) -> torch.Tensor:
    """Build (1, len(values), HEAD_DIM) where token i has every entry equal to values[i]."""
    return (
        torch.tensor(values, dtype=torch.float32, device=_DEVICE)
        .reshape(1, -1, 1)
        .expand(1, -1, _HEAD_DIM)
        .contiguous()
    )


def _cached_token_values(cached: torch.Tensor) -> list[float]:
    """Read scalar token values from a (B, S, H, D) cache tensor along S."""
    return cached[0, :, 0, 0].tolist()


def _block_state(transformer: _FakeTransformer) -> RollingKVCacheBlockState:
    hook = transformer.attn._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK)
    if hook.block_state_manager._current_context is None:
        hook.block_state_manager.set_context("inference")
    return hook.block_state_manager.get_state()


# ---------- State-class unit tests ----------


class TestStateClasses(unittest.TestCase):
    def test_rolling_kv_cache_state_defaults(self):
        state = RollingKVCacheState()
        self.assertTrue(state.should_update_cache)
        self.assertEqual(state.write_mode, "append")
        self.assertIsNone(state.absolute_token_offset)

    def test_rolling_kv_cache_state_reset(self):
        state = RollingKVCacheState()
        state.should_update_cache = False
        state.configure_cache_write(write_mode="overwrite", absolute_token_offset=8)
        state.reset()
        self.assertTrue(state.should_update_cache)
        self.assertEqual(state.write_mode, "append")
        self.assertIsNone(state.absolute_token_offset)

    def test_configure_cache_write_rejects_offset_in_append_mode(self):
        with self.assertRaises(ValueError):
            RollingKVCacheState().configure_cache_write(write_mode="append", absolute_token_offset=4)

    def test_configure_cache_write_requires_offset_in_overwrite_mode(self):
        with self.assertRaises(ValueError):
            RollingKVCacheState().configure_cache_write(write_mode="overwrite")

    def test_block_state_reset(self):
        state = RollingKVCacheBlockState()
        state.cached_key = torch.randn(1, 4, 2, _HEAD_DIM)
        state.cached_value = torch.randn(1, 4, 2, _HEAD_DIM)
        state.cache_start_token_offset = 16
        state.reset()
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)
        self.assertEqual(state.cache_start_token_offset, 0)


# ---------- Rotary helper unit tests ----------


class TestRotaryEmb(unittest.TestCase):
    def setUp(self):
        self.processor = RollingKVAttentionProcessor()

    def test_output_shape_and_dtype_preserved(self):
        x = torch.randn(1, 4, 2, 16, dtype=torch.bfloat16)
        freqs_cos = torch.ones(1, 4, 1, 16)
        freqs_sin = torch.zeros(1, 4, 1, 16)

        out = self.processor.apply_rotary_emb(x, freqs_cos, freqs_sin)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        torch.testing.assert_close(out, x)

    def test_matches_complex_reference(self):
        x = torch.randn(1, 4, 2, 16, dtype=torch.bfloat16)
        freqs_cos = torch.randn(1, 4, 1, 16, dtype=torch.float32)
        freqs_sin = torch.randn(1, 4, 1, 16, dtype=torch.float32)

        expected = torch.view_as_real(
            torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
            * torch.complex(freqs_cos[..., 0::2].to(torch.float64), freqs_sin[..., 1::2].to(torch.float64))
        ).flatten(-2)
        expected = expected.to(x.dtype)

        out = self.processor.apply_rotary_emb(x, freqs_cos, freqs_sin)

        torch.testing.assert_close(out, expected)


# ---------- Cache mechanics, expressed as token values ----------


class TestRollingKVCacheMechanics(unittest.TestCase):
    def test_first_pass_caches_input_tokens_verbatim(self):
        transformer = _make_transformer()

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))

        state = _block_state(transformer)
        self.assertEqual(_cached_token_values(state.cached_key), [0.0, 1.0, 2.0])
        self.assertEqual(_cached_token_values(state.cached_value), [0.0, 1.0, 2.0])
        self.assertEqual(state.cache_start_token_offset, 0)

    def test_second_pass_appends_to_existing_cache(self):
        transformer = _make_transformer()

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))
            transformer(_ramp([3, 4, 5]))

        state = _block_state(transformer)
        self.assertEqual(_cached_token_values(state.cached_key), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(state.cache_start_token_offset, 0)

    def test_window_trims_oldest_tokens_and_advances_start_offset(self):
        transformer = _make_transformer(window_size=4)

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))
            transformer(_ramp([3, 4, 5]))

        # Cache reached length 6, window keeps the most recent 4. Start offset moves to 2 so that
        # `cache_start_token_offset + cached_len` is still the absolute end of the stream (=6).
        state = _block_state(transformer)
        self.assertEqual(_cached_token_values(state.cached_key), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(state.cache_start_token_offset, 2)

    def test_should_update_cache_false_freezes_cache(self):
        transformer = _make_transformer()
        cache_state = get_rolling_kv_cache_state(transformer)

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))

        cache_state.should_update_cache = False
        with torch.no_grad():
            transformer(_ramp([3, 4, 5]))

        # Forward still ran — but the new chunk did NOT enter the cache.
        state = _block_state(transformer)
        self.assertEqual(_cached_token_values(state.cached_key), [0.0, 1.0, 2.0])

    def test_overwrite_replaces_suffix_at_absolute_offset(self):
        transformer = _make_transformer()
        cache_state = get_rolling_kv_cache_state(transformer)

        # Three chunks → cache holds tokens 0..8 at absolute positions 0..8.
        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))
            transformer(_ramp([3, 4, 5]))
            transformer(_ramp([6, 7, 8]))

        # Rewind to absolute offset 3 and replace the [3,4,5] + [6,7,8] suffix with new values.
        cache_state.configure_cache_write(write_mode="overwrite", absolute_token_offset=3)
        try:
            with torch.no_grad():
                transformer(_ramp([90, 91, 92]))
        finally:
            cache_state.clear_cache_write()

        state = _block_state(transformer)
        # Tokens at absolute 6..8 are gone; new tokens land at absolute 3..5.
        self.assertEqual(_cached_token_values(state.cached_key), [0.0, 1.0, 2.0, 90.0, 91.0, 92.0])
        self.assertEqual(state.cache_start_token_offset, 0)

    def test_overwrite_at_cache_start_drops_entire_prefix(self):
        transformer = _make_transformer()
        cache_state = get_rolling_kv_cache_state(transformer)

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))
            transformer(_ramp([3, 4, 5]))

        cache_state.configure_cache_write(write_mode="overwrite", absolute_token_offset=0)
        try:
            with torch.no_grad():
                transformer(_ramp([7, 8]))
        finally:
            cache_state.clear_cache_write()

        state = _block_state(transformer)
        self.assertEqual(_cached_token_values(state.cached_key), [7.0, 8.0])
        self.assertEqual(state.cache_start_token_offset, 0)

    def test_overwrite_past_cache_end_raises(self):
        transformer = _make_transformer()
        cache_state = get_rolling_kv_cache_state(transformer)

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))

        # Cache covers absolute positions 0..2 (cache_end=3). Asking to overwrite at 5 leaves a hole.
        cache_state.configure_cache_write(write_mode="overwrite", absolute_token_offset=5)
        with self.assertRaisesRegex(ValueError, "beyond the retained cache prefix"):
            with torch.no_grad():
                transformer(_ramp([9, 10]))

    def test_cache_context_isolates_cond_and_uncond(self):
        transformer = _make_transformer()

        with torch.no_grad(), transformer.cache_context("cond"):
            transformer(_ramp([0, 1, 2]))
            self.assertEqual(_cached_token_values(_block_state(transformer).cached_key), [0.0, 1.0, 2.0])

        with torch.no_grad(), transformer.cache_context("uncond"):
            transformer(_ramp([10, 20, 30]))
            self.assertEqual(
                _cached_token_values(_block_state(transformer).cached_key), [10.0, 20.0, 30.0]
            )

        # Re-entering "cond" sees the cond cache untouched, then appends.
        with torch.no_grad(), transformer.cache_context("cond"):
            transformer(_ramp([3, 4, 5]))
            self.assertEqual(
                _cached_token_values(_block_state(transformer).cached_key),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            )

    def test_reset_stateful_hooks_clears_cache(self):
        transformer = _make_transformer()

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))
        self.assertIsNotNone(_block_state(transformer).cached_key)

        transformer._diffusers_hook.reset_stateful_hooks()

        state = _block_state(transformer)
        self.assertIsNone(state.cached_key)
        self.assertIsNone(state.cached_value)
        self.assertEqual(state.cache_start_token_offset, 0)

    def test_batch_size_mismatch_raises(self):
        transformer = _make_transformer()

        with torch.no_grad():
            transformer(_ramp([0, 1, 2]))  # cached batch = 1

        # New chunk with batch=2 by stacking the same ramp; cached cache was batch=1.
        chunk = _ramp([3, 4, 5]).expand(2, -1, -1).contiguous()
        with self.assertRaisesRegex(ValueError, "batch size mismatch"):
            with torch.no_grad():
                transformer(chunk)

    def test_cache_mixin_enable_disable_cache(self):
        transformer = _FakeTransformer().to(_DEVICE).eval()

        logging.getLogger("diffusers.hooks.rolling_kv_cache").setLevel(logging.ERROR)
        transformer.enable_cache(RollingKVCacheConfig(window_size=4))
        logging.getLogger("diffusers.hooks.rolling_kv_cache").setLevel(logging.WARNING)

        self.assertTrue(transformer.is_cache_enabled)
        self.assertIsNotNone(transformer.attn._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK))
        self.assertIsNotNone(get_rolling_kv_cache_state(transformer))

        transformer.disable_cache()

        self.assertFalse(transformer.is_cache_enabled)
        self.assertIsNone(transformer.attn._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK))


# ---------- Integration: real Wan attention selection + warning ----------


class TestApplyRollingKVCacheOnWan(unittest.TestCase):
    """One sanity check that the duck-typed self-attn detection actually picks WanAttention."""

    def test_hooks_attach_to_self_attention_only(self):
        from diffusers import WanTransformer3DModel
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        config = {
            "patch_size": [1, 2, 2],
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 32,
            "freq_dim": 32,
            "ffn_dim": 64,
            "num_layers": 2,
            "cross_attn_norm": False,
            "qk_norm": "rms_norm_across_heads",
            "eps": 1e-6,
            "image_dim": None,
            "added_kv_proj_dim": None,
            "rope_max_seq_len": 32,
        }
        torch.manual_seed(0)
        transformer = WanTransformer3DModel.from_config(config).to(_DEVICE).eval()
        apply_rolling_kv_cache(transformer, RollingKVCacheConfig(window_size=-1))

        blocks = [m for m in transformer.modules() if isinstance(m, WanTransformerBlock)]
        self.assertEqual(len(blocks), config["num_layers"])
        for block in blocks:
            self.assertIsNotNone(block.attn1._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK))
            if hasattr(block.attn2, "_diffusers_hook"):
                self.assertIsNone(block.attn2._diffusers_hook.get_hook(_ROLLING_KV_CACHE_HOOK))

    def test_warns_when_attaching_to_untested_class(self):
        # _IdentitySelfAttention is not in the tested set, so apply_rolling_kv_cache must warn.
        with self.assertLogs("diffusers.hooks.rolling_kv_cache", level="WARNING") as captured:
            apply_rolling_kv_cache(_IdentitySelfAttention(), RollingKVCacheConfig(window_size=-1))
        self.assertTrue(
            any("_IdentitySelfAttention" in msg and "untested" in msg for msg in captured.output),
            captured.output,
        )


if __name__ == "__main__":
    unittest.main()
