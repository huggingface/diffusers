# coding=utf-8
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
import unittest

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import (
    _CAN_USE_FLASH_ATTN,
    AttentionBackendName,
    _pack_qkv,
    dispatch_attention_fn,
)


# A mask with non-contiguous valid tokens (gaps in the middle of each row).
# Row 0: positions 0-2 valid, 3-4 invalid, 5-9 valid  → 8 valid tokens
# Row 1: position  0   valid, 1-3 invalid, 4-9 valid  → 7 valid tokens
_NON_PREFIX_MASK = torch.tensor(
    [
        [True, True, True, False, False, True, True, True, True, True],
        [True, False, False, False, True, True, True, True, True, True],
    ],
    dtype=torch.bool,
)


def _make_qkv(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32):
    """Return reproducible (batch_size, seq_len, num_heads, head_dim) Q/K/V tensors."""
    g = torch.Generator().manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, generator=g, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, generator=g, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, generator=g, dtype=dtype)
    return q, k, v


class TestPackQkv(unittest.TestCase):
    """_pack_qkv: shapes, cu_seqlens, and round-trip via unpack()."""

    def test_kv_packed_q_full_length(self):
        """attn_mask is a KV-validity mask: K/V are packed, Q is kept at full length."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 16
        q, k, v = _make_qkv(batch_size, seq_len, num_heads, head_dim)
        packed = _pack_qkv(q, k, v, _NON_PREFIX_MASK)

        num_valid_tokens = int(_NON_PREFIX_MASK.sum())
        # K and V must be packed down to valid tokens only
        self.assertEqual(packed.key.shape, (num_valid_tokens, num_heads, head_dim))
        self.assertEqual(packed.value.shape, (num_valid_tokens, num_heads, head_dim))
        # Q must remain full-length (flattened but not filtered)
        self.assertEqual(packed.query.shape, (batch_size * seq_len, num_heads, head_dim))
        self.assertEqual(packed.seq_len_q, seq_len)
        self.assertEqual(packed.cu_seqlens_q[-1].item(), batch_size * seq_len)
        self.assertEqual(packed.cu_seqlens_k[-1].item(), num_valid_tokens)

    def test_cu_seqlens_reflect_valid_counts_not_positions(self):
        """Non-prefix mask: cu_seqlens counts valid tokens per batch item, ignoring gaps."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 16
        q, k, v = _make_qkv(batch_size, seq_len, num_heads, head_dim)
        packed = _pack_qkv(q, k, v, _NON_PREFIX_MASK)

        # Row 0 has 8 valid tokens; row 1 has 7 valid tokens (see _NON_PREFIX_MASK).
        valid_per_item = _NON_PREFIX_MASK.sum(dim=-1)
        self.assertEqual(packed.cu_seqlens_k[1].item(), valid_per_item[0].item())
        self.assertEqual(packed.cu_seqlens_k[2].item(), int(valid_per_item.sum()))
        self.assertEqual(packed.max_seqlen_k, int(valid_per_item.max()))

    def test_unpack_reshapes_full_length_output(self):
        """unpack() with indices_q=None just reshapes the flat output back to (batch_size, seq_len, ...)."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 16
        q, k, v = _make_qkv(batch_size, seq_len, num_heads, head_dim)
        packed = _pack_qkv(q, k, v, _NON_PREFIX_MASK)

        # Q is not packed, so a fake attention output matching the full flattened Q shape
        # should round-trip back to the original padded layout unchanged.
        recovered = packed.unpack(packed.query)
        self.assertEqual(recovered.shape, (batch_size, seq_len, num_heads, head_dim))
        self.assertTrue(torch.allclose(recovered, q))

    def test_cross_attn_q_is_not_packed(self):
        """Cross-attention (seq_q != seq_kv): Q remains full-length instead of being packed."""
        batch_size, seq_len_q, seq_len_kv, num_heads, head_dim = 2, 5, 10, 2, 8
        # _NON_PREFIX_MASK has shape (2, 10) and applies to KV tokens only
        q = torch.randn(batch_size, seq_len_q, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim)

        packed = _pack_qkv(q, k, v, _NON_PREFIX_MASK)

        self.assertEqual(packed.query.shape, (batch_size * seq_len_q, num_heads, head_dim))
        self.assertEqual(packed.seq_len_q, seq_len_q)


@unittest.skipUnless(_CAN_USE_FLASH_ATTN, "flash-attn is required for these tests")
class TestFlashAttentionWithMask(unittest.TestCase):
    """Flash attention backend must produce results consistent with the SDPA reference when attn_mask is given."""

    def _sdpa_ref(self, q, k, v, bool_mask_2d):
        """SDPA reference: converts a 2D bool mask to an additive float mask and runs SDPA."""
        # Additive mask convention: 0.0 for positions to attend to, -inf for positions to ignore.
        additive_mask = torch.zeros_like(bool_mask_2d, dtype=q.dtype)
        additive_mask = additive_mask.masked_fill(~bool_mask_2d, float("-inf"))
        additive_mask = additive_mask[:, None, None, :]  # (batch_size, 1, 1, seq_len_kv)
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)
        return out.permute(0, 2, 1, 3)

    def test_non_prefix_mask_matches_sdpa_reference(self):
        """Non-prefix mask: FLASH backend output must match SDPA reference."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 32
        device = torch.device("cuda")
        q, k, v = (
            t.to(device=device, dtype=torch.float16) for t in _make_qkv(batch_size, seq_len, num_heads, head_dim)
        )
        mask = _NON_PREFIX_MASK.to(device)

        ref = self._sdpa_ref(q, k, v, mask)
        out = dispatch_attention_fn(q, k, v, attn_mask=mask, backend=AttentionBackendName.FLASH)

        self.assertTrue(torch.allclose(ref, out, atol=1e-2), f"Max diff: {(ref - out).abs().max():.2e}")

    def test_all_valid_mask_equals_no_mask(self):
        """All-True mask must produce the same output as passing no mask at all."""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 2, 32
        device = torch.device("cuda")
        q, k, v = (
            t.to(device=device, dtype=torch.float16) for t in _make_qkv(batch_size, seq_len, num_heads, head_dim)
        )
        all_valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        out_masked = dispatch_attention_fn(q, k, v, attn_mask=all_valid_mask, backend=AttentionBackendName.FLASH)
        out_no_mask = dispatch_attention_fn(q, k, v, attn_mask=None, backend=AttentionBackendName.FLASH)

        self.assertTrue(torch.allclose(out_masked, out_no_mask, atol=1e-3))

    def test_4d_bool_mask_equivalent_to_2d(self):
        """4D bool mask (batch_size, 1, 1, seq_len) must normalize to the same result as the 2D mask."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 32
        device = torch.device("cuda")
        q, k, v = (
            t.to(device=device, dtype=torch.float16) for t in _make_qkv(batch_size, seq_len, num_heads, head_dim)
        )
        mask = _NON_PREFIX_MASK.to(device)

        out_2d = dispatch_attention_fn(q, k, v, attn_mask=mask, backend=AttentionBackendName.FLASH)
        out_4d = dispatch_attention_fn(q, k, v, attn_mask=mask[:, None, None, :], backend=AttentionBackendName.FLASH)

        self.assertTrue(torch.allclose(out_2d, out_4d, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
