"""Regression test for the sage-attention + ring-attention crash.

Issue #13506 reports that running the Sage attention kernel inside the ring
attention code path (TemplatedRingAttention) raises an illegal CUDA memory
access. The root cause is that ``funcol.all_gather_tensor(...).chunk(...)``
returns tensor views whose ``storage_offset`` is non-zero, but the Sage
kernels call ``tensor.data_ptr()`` and assume the offset is zero.

The fix in ``_sage_attention_forward_op`` and ``_sage_attention_hub_forward_op``
materializes contiguous copies of the q/k/v tensors before invoking the
kernel, so non-zero ``storage_offset`` views no longer corrupt the data
pointer used by the kernel.

This test simulates the failing shape on CPU tensors and monkey-patches
``sageattn`` to capture the tensor the kernel would have received, then
asserts the tensor's ``storage_offset`` is zero. We do not need a real
CUDA / Sage install to validate the fix, only that the input was made
contiguous.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from diffusers.models.attention_dispatch import (
    _HUB_KERNELS_REGISTRY,
    AttentionBackendName,
    _sage_attention_forward_op,
    _sage_attention_hub_forward_op,
)


def _make_chunked_view(batch: int, seq: int, heads: int, dim: int, world_size: int = 2) -> torch.Tensor:
    """Build a tensor view that mimics a single chunk of an all_gather result.

    A real ring-attention chunk has the same data layout as a narrow view of
    a larger all-gathered buffer: same storage, just a non-zero storage_offset
    and a different shape. We construct it via ``cat + chunk`` so the offset
    and data match what ``funcol.all_gather_tensor(...).chunk(world_size)``
    would produce.
    """
    full = torch.randn(batch * world_size, seq, heads, dim, dtype=torch.float32)
    chunks = full.chunk(world_size)
    return chunks[0]


class SageAttentionRingAttentionStorageOffsetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q = _make_chunked_view(batch=1, seq=8, heads=2, dim=4, world_size=2)
        self.k = _make_chunked_view(batch=1, seq=8, heads=2, dim=4, world_size=2)
        self.v = _make_chunked_view(batch=1, seq=8, heads=2, dim=4, world_size=2)
        # Sanity: the helper above really does produce non-contiguous views.
        for name, t in [("q", self.q), ("k", self.k), ("v", self.v)]:
            assert t.storage_offset() != 0, f"test fixture wrong: {name} has zero storage_offset"
            assert not t.is_contiguous(), f"test fixture wrong: {name} is already contiguous"

    def test_sage_forward_op_makes_inputs_contiguous(self):
        captured = {}

        def fake_sageattn(q, k, v, tensor_layout, is_causal, sm_scale, return_lse):
            captured["q"] = q
            captured["k"] = k
            captured["v"] = v
            return q.new_zeros(q.shape)

        with patch("diffusers.models.attention_dispatch.sageattn", side_effect=fake_sageattn):
            _sage_attention_forward_op(None, self.q, self.k, self.v)

        for name in ("q", "k", "v"):
            t = captured[name]
            self.assertEqual(
                t.storage_offset(),
                0,
                f"{name} still has non-zero storage_offset after the fix",
            )
            self.assertTrue(t.is_contiguous(), f"{name} is not contiguous after the fix")
            # Make sure the values themselves are unchanged
            self.assertTrue(torch.equal(t, getattr(self, name)))

    def test_sage_hub_forward_op_makes_inputs_contiguous(self):
        captured = {}

        def fake_kernel(q, k, v, tensor_layout, is_causal, sm_scale, return_lse):
            captured["q"] = q
            captured["k"] = k
            captured["v"] = v
            return q.new_zeros(q.shape)

        # Swap the hub registry entry for the duration of the call. The op
        # reads ``_HUB_KERNELS_REGISTRY[SAGE_HUB].kernel_fn`` lazily so this
        # is sufficient.
        original = _HUB_KERNELS_REGISTRY[AttentionBackendName.SAGE_HUB]
        try:
            _HUB_KERNELS_REGISTRY[AttentionBackendName.SAGE_HUB] = SimpleNamespace(kernel_fn=fake_kernel)
            _sage_attention_hub_forward_op(None, self.q, self.k, self.v)
        finally:
            _HUB_KERNELS_REGISTRY[AttentionBackendName.SAGE_HUB] = original

        for name in ("q", "k", "v"):
            t = captured[name]
            self.assertEqual(
                t.storage_offset(),
                0,
                f"{name} still has non-zero storage_offset after the fix",
            )
            self.assertTrue(t.is_contiguous(), f"{name} is not contiguous after the fix")
            self.assertTrue(torch.equal(t, getattr(self, name)))


if __name__ == "__main__":
    unittest.main()
