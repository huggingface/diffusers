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

from diffusers.pipelines.kolors.text_encoder import ChatGLMConfig, CoreAttention

from ...testing_utils import torch_device


class TestKolorsCoreAttention:
    """
    `CoreAttention` computes raw attention scores. On MPS it must not route them through
    `baddbmm(input=torch.empty(...), beta=0)`, because MPS does not honour the documented
    "input is ignored when beta=0" contract and NaN from the uninitialised buffer can reach
    the scores. See https://github.com/huggingface/diffusers/pull/14459.
    """

    def get_attention(self):
        config = ChatGLMConfig(
            hidden_size=256,
            num_attention_heads=8,
            kv_channels=32,
            multi_query_attention=False,
            attention_softmax_in_fp32=True,
        )
        return CoreAttention(config, layer_number=1)

    def get_inputs(self, device, dtype=torch.float32, seq_len=256, batch=2, heads=8, head_dim=32):
        torch.manual_seed(0)
        shape = (seq_len, batch, heads, head_dim)
        return tuple(torch.randn(shape, dtype=dtype).to(device) for _ in range(3))

    def test_scores_match_cpu_reference(self):
        # The device-specific score path must stay numerically equivalent to the CPU path.
        attention = self.get_attention()
        query, key, value = self.get_inputs("cpu")

        with torch.no_grad():
            expected = attention(query, key, value, None)
            actual = attention(query.to(torch_device), key.to(torch_device), value.to(torch_device), None)

        torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(torch_device != "mps", reason="guards an MPS-specific baddbmm contract violation")
    def test_scores_finite_with_dirty_allocator(self):
        # Free NaN-filled blocks of exactly the scores shape so the allocator can hand those
        # pages to the score computation, then assert the output stays finite.
        attention = self.get_attention()
        seq_len, batch, heads = 256, 2, 8
        query, key, value = self.get_inputs(torch_device, dtype=torch.float16, seq_len=seq_len)

        scores_shape = (batch * heads, seq_len, seq_len)
        for _ in range(4):
            junk = torch.full(scores_shape, float("nan"), device=torch_device, dtype=torch.float16)
            del junk

        with torch.no_grad():
            output = attention(query, key, value, None)

        assert torch.isfinite(output).all(), "NaN reached the Kolors attention output on MPS"
