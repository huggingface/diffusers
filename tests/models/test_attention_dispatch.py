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

from diffusers.models.attention_dispatch import _cudnn_attention_forward_op

from ..testing_utils import assert_tensors_close, is_attention, require_torch_gpu, torch_device


@is_attention
@require_torch_gpu
class TestCudnnAttentionForwardOp:
    @pytest.mark.parametrize("mask_type", ["partial", "fully_masked_row"])
    def test_boolean_attn_mask_matches_sdpa(self, mask_type):
        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 64
        torch.manual_seed(0)

        # the forward op takes `(batch_size, seq_len, num_heads, head_dim)`
        query, key, value = (
            torch.randn(batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.bfloat16)
            for _ in range(3)
        )
        attn_mask = torch.ones(batch_size, num_heads, seq_len, seq_len, device=torch_device, dtype=torch.bool)
        if mask_type == "partial":
            attn_mask[..., 3, 5:] = False
        else:
            attn_mask[..., 7, :] = False

        out = _cudnn_attention_forward_op(None, query, key, value, attn_mask=attn_mask, _save_ctx=False)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attn_mask
        ).transpose(1, 2)

        assert_tensors_close(out, expected, atol=1e-2, rtol=1e-2, msg=f"cuDNN forward op with {mask_type} mask")
