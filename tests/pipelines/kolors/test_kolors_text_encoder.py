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
    `CoreAttention` computes attention through `scaled_dot_product_attention`; the module
    used to carry a second, manual attention path gated on torch < 2 that was unreachable
    on any supported torch (https://github.com/huggingface/diffusers/issues/14624). These
    tests pin the behaviour of the remaining path on every backend so its removal, and any
    future rework, stay observable.
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

    def get_inputs(self, device, dtype=torch.float32, seq_len=64, batch=2, heads=8, head_dim=32):
        torch.manual_seed(0)
        shape = (seq_len, batch, heads, head_dim)
        return tuple(torch.randn(shape, dtype=dtype).to(device) for _ in range(3))

    @pytest.mark.parametrize("masked", [False, True])
    def test_output_matches_cpu_reference(self, masked):
        # The device path must stay numerically equivalent to the CPU path, for both the
        # causal (mask=None) branch and the explicit-mask branch of forward.
        attention = self.get_attention()
        query, key, value = self.get_inputs("cpu")
        seq_len, batch = query.shape[0], query.shape[1]

        if masked:
            torch.manual_seed(1)
            # ChatGLM convention: True marks positions that must NOT be attended to.
            attention_mask = torch.rand(batch, 1, seq_len, seq_len) < 0.25
            attention_mask[..., 0] = False  # keep at least one visible key per query row
        else:
            attention_mask = None

        with torch.no_grad():
            expected = attention(query, key, value, attention_mask)
            actual = attention(
                query.to(torch_device),
                key.to(torch_device),
                value.to(torch_device),
                attention_mask.to(torch_device) if attention_mask is not None else None,
            )

        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
