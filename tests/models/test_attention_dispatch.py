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

import pytest
import torch

from diffusers.models.attention_dispatch import (
    _cudnn_attention_backward_op,
    _cudnn_attention_forward_op,
    _native_flash_attention_backward_op,
    _native_flash_attention_forward_op,
)

from ..testing_utils import assert_tensors_close, is_attention, require_torch_gpu


class _TemplatedAttentionOp(torch.autograd.Function):
    """Minimal driver for a (forward_op, backward_op) pair, mirroring the context parallel wrappers."""

    @staticmethod
    def forward(ctx, query, key, value, forward_op, backward_op):
        ctx.backward_op = backward_op
        # The context parallel wrappers always request the lse.
        out, _ = forward_op(ctx, query, key, value, return_lse=True)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_query, grad_key, grad_value = ctx.backward_op(ctx, grad_out)
        return grad_query, grad_key, grad_value, None, None


@is_attention
@require_torch_gpu
@pytest.mark.parametrize(
    "forward_op,backward_op",
    [
        (_cudnn_attention_forward_op, _cudnn_attention_backward_op),
        (_native_flash_attention_forward_op, _native_flash_attention_backward_op),
    ],
    ids=["cudnn", "native_flash"],
)
def test_templated_attention_op_gradients(forward_op, backward_op):
    """Gradients from the templated forward/backward op pairs must match an eager fp32 reference.

    `num_heads == seq_len` on purpose so that a (B, S, H, D) / (B, H, S, D) layout mix-up cannot be
    hidden by a shape mismatch. See https://github.com/huggingface/diffusers/issues/14338.
    """
    torch.manual_seed(0)
    batch_size, seq_len, num_heads, head_dim = 2, 16, 16, 64
    shape = (batch_size, seq_len, num_heads, head_dim)
    query, key, value, grad_out = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(4)
    )

    ref_query, ref_key, ref_value = (
        x.detach().float().transpose(1, 2).requires_grad_(True) for x in (query, key, value)
    )
    ref_out = torch.nn.functional.scaled_dot_product_attention(ref_query, ref_key, ref_value)
    ref_out.backward(grad_out.detach().float().transpose(1, 2))

    out = _TemplatedAttentionOp.apply(query, key, value, forward_op, backward_op)
    out.backward(grad_out)

    assert_tensors_close(out.float(), ref_out.transpose(1, 2), atol=1e-2, rtol=1e-2, msg="forward output")
    for name, actual, expected in (
        ("query", query.grad, ref_query.grad),
        ("key", key.grad, ref_key.grad),
        ("value", value.grad, ref_value.grad),
    ):
        assert_tensors_close(
            actual.float(), expected.transpose(1, 2), atol=1e-1, rtol=1e-1, msg=f"gradient w.r.t. {name}"
        )
