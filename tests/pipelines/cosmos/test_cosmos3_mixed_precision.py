# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from diffusers.pipelines.cosmos.mixed_precision import (
    Cosmos3MixedPrecisionConfig,
    apply_cosmos3_mixed_precision_step,
    reset_cosmos3_mixed_precision,
)


class _FakeModelOptFp8Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[4.0, -2.0], [1.0, 3.0]], dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.bias = None
        self.input_quantizer = SimpleNamespace(is_enabled=True)
        self.weight_quantizer = SimpleNamespace(_scale=torch.tensor(0.25), is_enabled=True)
        self._should_run_real_quant_gemm = True

    def forward(self, inputs):
        # Stand in for ModelOpt's native W8A8 GEMM.
        quantized_inputs = inputs.to(torch.float8_e4m3fn).to(inputs.dtype)
        dense_weight = self.weight.to(inputs.dtype) * self.weight_quantizer._scale.to(inputs.dtype)
        return F.linear(quantized_inputs, dense_weight, self.bias)


class _FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = _FakeModelOptFp8Linear()
        self.add_q_proj = _FakeModelOptFp8Linear()


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer()])


class Cosmos3MixedPrecisionConfigTests(unittest.TestCase):
    def test_default_50_step_schedule_matches_vllm_omni(self):
        config = Cosmos3MixedPrecisionConfig(format="fp8")
        selected = [i for i in range(50) if config.use_high_precision(i, 50)]
        self.assertEqual(selected, [0, 1, 2, 47, 48, 49])
        self.assertEqual(config.precision_name(0, 50), "W8A16")
        self.assertEqual(config.precision_name(25, 50), "W8A8")

    def test_asymmetric_and_overlap_boundaries(self):
        cases = [
            (2, 4, 10, [0, 1, 6, 7, 8, 9]),
            (0, 2, 7, [5, 6]),
            (2, 0, 7, [0, 1]),
            (0, 0, 7, []),
            (4, 4, 7, list(range(7))),
        ]
        for first_steps, last_steps, num_steps, selected in cases:
            with self.subTest(first=first_steps, last=last_steps, n=num_steps):
                config = Cosmos3MixedPrecisionConfig(format="fp8", first_steps=first_steps, last_steps=last_steps)
                self.assertEqual(
                    [i for i in range(num_steps) if config.use_high_precision(i, num_steps)],
                    selected,
                )

    def test_one_step_keeps_base_precision(self):
        config = Cosmos3MixedPrecisionConfig(format="fp8", first_steps=1, last_steps=1)
        self.assertFalse(config.use_high_precision(0, 1))

    def test_disabled_format_is_noop(self):
        config = Cosmos3MixedPrecisionConfig.from_kwargs(mixed_precision_format="none")
        self.assertFalse(config.enabled)
        transformer = _Transformer()
        trace = []
        inputs = torch.tensor([[1.1, -0.7]], dtype=torch.bfloat16)
        expected = transformer.layers[0].self_attn.to_q(inputs)
        name = apply_cosmos3_mixed_precision_step(transformer, config, 0, 10, trace=trace)
        self.assertEqual(name, "base")
        torch.testing.assert_close(transformer.layers[0].self_attn.to_q(inputs), expected)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            Cosmos3MixedPrecisionConfig.from_kwargs(mixed_precision_format="nvfp4")
        with self.assertRaises(ValueError):
            Cosmos3MixedPrecisionConfig.from_kwargs(mixed_precision_reasoner_policy="fp16")
        with self.assertRaises(TypeError):
            Cosmos3MixedPrecisionConfig.from_kwargs(mixed_precision_first_steps=-1)

    def test_dispatches_generation_w8a16_edges_and_w8a8_middle(self):
        config = Cosmos3MixedPrecisionConfig(format="fp8", first_steps=1, last_steps=1)
        transformer = _Transformer()
        generation = transformer.layers[0].self_attn.to_q
        reasoner = transformer.layers[0].self_attn.add_q_proj
        inputs = torch.tensor([[1.1, -0.7]], dtype=torch.bfloat16)
        expected_w8a8 = generation(inputs)
        expected_w8a16 = F.linear(
            inputs,
            generation.weight.to(inputs.dtype) * generation.weight_quantizer._scale.to(inputs.dtype),
        )
        self.assertFalse(torch.equal(expected_w8a8, expected_w8a16))

        trace = []
        apply_cosmos3_mixed_precision_step(transformer, config, 0, 5, trace=trace)
        torch.testing.assert_close(generation(inputs), expected_w8a16)
        torch.testing.assert_close(reasoner(inputs), expected_w8a16)

        apply_cosmos3_mixed_precision_step(transformer, config, 2, 5, trace=trace)
        torch.testing.assert_close(generation(inputs), expected_w8a8)
        # The default policy keeps the reasoner in high precision on every step.
        torch.testing.assert_close(reasoner(inputs), expected_w8a16)

        apply_cosmos3_mixed_precision_step(transformer, config, 4, 5, trace=trace)
        torch.testing.assert_close(generation(inputs), expected_w8a16)
        self.assertEqual(trace, ["W8A16", "W8A8", "W8A16"])

        reset_cosmos3_mixed_precision(transformer, config)
        torch.testing.assert_close(generation(inputs), expected_w8a8)
        torch.testing.assert_close(reasoner(inputs), expected_w8a8)

    def test_reasoner_can_use_base_precision(self):
        config = Cosmos3MixedPrecisionConfig(
            format="fp8",
            first_steps=0,
            last_steps=0,
            reasoner_policy="base_precision",
        )
        transformer = _Transformer()
        reasoner = transformer.layers[0].self_attn.add_q_proj
        inputs = torch.tensor([[1.1, -0.7]], dtype=torch.bfloat16)
        expected_w8a8 = reasoner(inputs)
        apply_cosmos3_mixed_precision_step(transformer, config, 2, 5)
        torch.testing.assert_close(reasoner(inputs), expected_w8a8)


if __name__ == "__main__":
    unittest.main()
