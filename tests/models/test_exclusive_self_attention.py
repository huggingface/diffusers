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

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    SlicedAttnProcessor,
    _apply_exclusive_self_attention,
)


class ExclusiveSelfAttentionTests(unittest.TestCase):
    def test_apply_exclusive_self_attention_orthogonalizes_output(self):
        torch.manual_seed(0)
        hidden_states = torch.randn(2, 3, 4)
        value = torch.randn(2, 3, 4)

        output = _apply_exclusive_self_attention(hidden_states, value)
        value_normalized = F.normalize(value, p=2, dim=-1, eps=1e-12)
        projection = (output * value_normalized).sum(dim=-1)

        assert torch.allclose(projection, torch.zeros_like(projection), atol=1e-5)
        assert output.dtype == hidden_states.dtype
        assert output.device == hidden_states.device

    def test_apply_exclusive_self_attention_handles_zero_values(self):
        hidden_states = torch.randn(2, 3, 4)
        value = torch.zeros_like(hidden_states)

        output = _apply_exclusive_self_attention(hidden_states, value)

        assert torch.isfinite(output).all()
        assert torch.allclose(output, hidden_states)

    def test_apply_exclusive_self_attention_skips_shape_mismatch(self):
        hidden_states = torch.randn(2, 3, 4)
        value = torch.randn(2, 2, 4)

        output = _apply_exclusive_self_attention(hidden_states, value)

        assert output is hidden_states

    def _get_attention_pair(self, processor_factory, fused=False, cross_attention_dim=None):
        torch.manual_seed(0)
        base = Attention(
            query_dim=8,
            cross_attention_dim=cross_attention_dim,
            heads=2,
            dim_head=4,
            dropout=0.0,
            bias=True,
            processor=AttnProcessor2_0() if fused else processor_factory(),
        )
        exclusive = Attention(
            query_dim=8,
            cross_attention_dim=cross_attention_dim,
            heads=2,
            dim_head=4,
            dropout=0.0,
            bias=True,
            exclusive_self_attention=True,
            processor=AttnProcessor2_0() if fused else processor_factory(),
        )
        exclusive.load_state_dict(base.state_dict())

        if fused:
            base.fuse_projections()
            exclusive.fuse_projections()
            base.set_processor(processor_factory())
            exclusive.set_processor(processor_factory())

        return base, exclusive

    def test_exclusive_self_attention_changes_self_attention_output(self):
        hidden_states = torch.randn(2, 4, 8)
        processor_factories = [
            ("eager", AttnProcessor, False),
            ("sliced", lambda: SlicedAttnProcessor(slice_size=2), False),
        ]
        if hasattr(F, "scaled_dot_product_attention"):
            processor_factories.extend(
                [
                    ("sdpa", AttnProcessor2_0, False),
                    ("fused", FusedAttnProcessor2_0, True),
                ]
            )

        for name, processor_factory, fused in processor_factories:
            with self.subTest(name=name):
                base, exclusive = self._get_attention_pair(processor_factory, fused=fused)

                base_output = base(hidden_states)
                exclusive_output = exclusive(hidden_states)

                assert base_output.shape == exclusive_output.shape
                assert torch.isfinite(exclusive_output).all()
                assert not torch.allclose(base_output, exclusive_output)

    def test_exclusive_self_attention_does_not_change_cross_attention_output(self):
        hidden_states = torch.randn(2, 4, 8)
        encoder_hidden_states = torch.randn(2, 4, 8)
        processor_factories = [
            ("eager", AttnProcessor, False),
            ("sliced", lambda: SlicedAttnProcessor(slice_size=2), False),
        ]
        if hasattr(F, "scaled_dot_product_attention"):
            processor_factories.extend(
                [
                    ("sdpa", AttnProcessor2_0, False),
                    ("fused", FusedAttnProcessor2_0, True),
                ]
            )

        for name, processor_factory, fused in processor_factories:
            with self.subTest(name=name):
                base, exclusive = self._get_attention_pair(processor_factory, fused=fused, cross_attention_dim=8)

                base_output = base(hidden_states, encoder_hidden_states=encoder_hidden_states)
                exclusive_output = exclusive(hidden_states, encoder_hidden_states=encoder_hidden_states)

                assert torch.allclose(base_output, exclusive_output)

    def test_basic_transformer_block_wires_exclusive_self_attention_to_self_attention_only(self):
        block = BasicTransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
            cross_attention_dim=8,
            exclusive_self_attention=True,
        )
        assert block.attn1.exclusive_self_attention
        assert not block.attn2.exclusive_self_attention

        block = BasicTransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
            cross_attention_dim=8,
            only_cross_attention=True,
            exclusive_self_attention=True,
        )
        assert not block.attn1.exclusive_self_attention
        assert not block.attn2.exclusive_self_attention

        block = BasicTransformerBlock(
            dim=8,
            num_attention_heads=2,
            attention_head_dim=4,
            double_self_attention=True,
            exclusive_self_attention=True,
        )
        assert block.attn1.exclusive_self_attention
        assert block.attn2.exclusive_self_attention
