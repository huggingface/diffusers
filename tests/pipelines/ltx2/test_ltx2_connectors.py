# Copyright 2026 The HuggingFace Team.
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

from diffusers.pipelines.ltx2.connectors import LTX2ConnectorTransformer1d

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class LTX2ConnectorRegisterLayoutTests(unittest.TestCase):
    """The connector must lay out its sequence exactly like the original LTX
    implementation (``ltx_core`` ``_replace_padded_with_learnable_registers``,
    also matched by ComfyUI): the valid tokens move to the front *in their
    original order*, and the tail is filled with the tiled learnable registers
    indexed by *absolute position*. The connector blocks apply RoPE, so any
    deviation (e.g. reversed token order) produces embeddings the DiT was
    never trained on.
    """

    num_registers = 4
    seq_len = 12
    num_heads = 2
    head_dim = 4

    def get_connector(self):
        # num_layers=0 keeps the forward to layout + final RMSNorm, so the
        # register layout can be checked exactly.
        return LTX2ConnectorTransformer1d(
            num_attention_heads=self.num_heads,
            attention_head_dim=self.head_dim,
            num_layers=0,
            num_learnable_registers=self.num_registers,
        ).eval()

    def get_inputs(self, valid_lengths):
        dim = self.num_heads * self.head_dim
        batch_size = len(valid_lengths)
        hidden_states = torch.randn(batch_size, self.seq_len, dim)
        # Left padding, like the Gemma tokenization in the LTX2 pipelines.
        binary_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.int64)
        for i, n in enumerate(valid_lengths):
            binary_mask[i, self.seq_len - n :] = 1
        additive_mask = (binary_mask - 1).to(hidden_states.dtype)
        additive_mask = additive_mask.reshape(batch_size, 1, 1, self.seq_len)
        additive_mask = additive_mask * torch.finfo(hidden_states.dtype).max
        return hidden_states, binary_mask, additive_mask

    def reference_layout(self, connector, hidden_states, binary_mask):
        # Reference semantics: front-align valid tokens (order preserved),
        # fill the tail with the register tile by absolute position.
        batch_size, seq_len, _ = hidden_states.shape
        registers = connector.learnable_registers.detach()
        tiled = registers.repeat(seq_len // self.num_registers, 1)
        expected = torch.empty_like(hidden_states)
        for i in range(batch_size):
            valid = hidden_states[i, binary_mask[i].bool()]
            expected[i, : valid.shape[0]] = valid
            expected[i, valid.shape[0] :] = tiled[valid.shape[0] :]
        # The forward ends with a non-affine RMSNorm.
        return expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + 1e-6)

    def check_layout(self, valid_lengths):
        connector = self.get_connector()
        hidden_states, binary_mask, additive_mask = self.get_inputs(valid_lengths)
        with torch.no_grad():
            output, _ = connector(hidden_states, additive_mask)
        expected = self.reference_layout(connector, hidden_states, binary_mask)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_register_layout_left_padded(self):
        self.check_layout([5])

    def test_register_layout_mixed_lengths_batch(self):
        # The pipelines concatenate negative and positive prompts of different
        # lengths into one batch; the layout must be computed per row.
        self.check_layout([5, 2])

    def test_register_layout_fully_valid(self):
        self.check_layout([self.seq_len])

    def test_register_layout_single_token(self):
        self.check_layout([1])
