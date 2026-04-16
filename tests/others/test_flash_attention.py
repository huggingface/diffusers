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
import pytest
import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import (
    _CAN_USE_FLASH_ATTN,
    AttentionBackendName,
    dispatch_attention_fn,
)


# A mask with non-contiguous valid tokens.
_NON_PREFIX_MASK = torch.tensor(
    [
        [True, True, True, False, False, True, True, True, True, True],
        [True, False, False, False, True, True, True, True, True, True],
    ],
    dtype=torch.bool,
)


def _make_qkv(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32):
    g = torch.Generator().manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, generator=g, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, generator=g, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, generator=g, dtype=dtype)
    return q, k, v


def _sdpa_ref(q, k, v, bool_mask_2d=None):
    if bool_mask_2d is not None:
        additive_mask = torch.zeros_like(bool_mask_2d, dtype=q.dtype)
        additive_mask = additive_mask.masked_fill(~bool_mask_2d, float("-inf"))
        additive_mask = additive_mask[:, None, None, :]  # (batch_size, 1, 1, seq_len_kv)
    else:
        additive_mask = None
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)
    return out.permute(0, 2, 1, 3)


@pytest.mark.skipif(not _CAN_USE_FLASH_ATTN, reason="flash-attn is required for these tests")
class TestFlashAttention:
    """Flash attention backend must produce results consistent with the SDPA reference when attn_mask is given."""

    def test_no_mask_matches_sdpa_reference(self):
        """FLASH backend output must match SDPA reference without any masking."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 32
        device = torch.device("cuda")
        q, k, v = (
            t.to(device=device, dtype=torch.float16) for t in _make_qkv(batch_size, seq_len, num_heads, head_dim)
        )
        ref = _sdpa_ref(q, k, v)
        out = dispatch_attention_fn(q, k, v, attn_mask=None, backend=AttentionBackendName.FLASH)

        assert torch.allclose(ref, out, atol=1e-2), f"Max diff: {(ref - out).abs().max():.2e}"

    def test_mask_matches_sdpa_reference(self):
        """FLASH backend output must match SDPA reference with attention mask."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 2, 32
        device = torch.device("cuda")
        q, k, v = (
            t.to(device=device, dtype=torch.float16) for t in _make_qkv(batch_size, seq_len, num_heads, head_dim)
        )
        mask = _NON_PREFIX_MASK.to(device)

        ref = _sdpa_ref(q, k, v, mask)
        out = dispatch_attention_fn(q, k, v, attn_mask=mask, backend=AttentionBackendName.FLASH)

        assert torch.allclose(ref, out, atol=1e-2), f"Max diff: {(ref - out).abs().max():.2e}"

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

        assert torch.allclose(out_2d, out_4d, atol=1e-3)
