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

import pytest
import torch

from diffusers.loaders.conversion import Conversion, Identity, ReorderChunks, Rule, Split


class TestCheckpointConversion:
    def test_zero123_projection_layout(self):
        from diffusers.loaders.conversion import get_conversion

        state = {"projection.weight": torch.randn(8, 16), "projection.bias": torch.randn(8)}
        conversion = get_conversion("CCProjection", {})
        original = conversion.to_original(state)
        assert set(original) == {"cc_projection.weight", "cc_projection.bias"}
        restored = conversion.to_diffusers(original)
        for key in state:
            torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)

    @pytest.mark.parametrize("key", ["", 2, None])
    def test_invalid_key_name(self, key):
        with pytest.raises(ValueError, match="nonempty strings"):
            Conversion(mapping={"original": key})

    def test_rename_and_key_swap_do_not_mutate_inputs(self):
        mapping = {"a": "b", "b": "a"}
        conversion = Conversion(mapping)
        mapping["a"] = "changed"
        original = {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}
        converted = conversion.to_diffusers(original)
        assert converted["b"] is original["a"]
        assert converted["a"] is original["b"]
        assert list(original) == ["a", "b"]
        assert original["a"].tolist() == [1, 2]
        restored = conversion.to_original(converted)
        for key in original:
            assert restored[key] is original[key]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("dim", [0, -1])
    def test_unequal_split_in_both_directions(self, dtype, dim):
        original = torch.arange(40, dtype=dtype).reshape(10, 4)
        if dim == -1:
            original = original.T
        conversion = Conversion(rules=(Rule(("qkv",), ("q", "k", "v"), Split((6, 2, 2), dim)),))
        converted = conversion.to_diffusers({"qkv": original})
        for name, expected in zip(("q", "k", "v"), original.split((6, 2, 2), dim=dim)):
            torch.testing.assert_close(converted[name], expected, rtol=0, atol=0)
        restored = conversion.to_original(converted)
        torch.testing.assert_close(restored["qkv"], original, rtol=0, atol=0)
        # Fresh destination weights can be exported without an earlier import.
        fresh = {"q": converted["q"] + 1, "k": converted["k"] + 2, "v": converted["v"] + 3}
        reimported = conversion.to_diffusers(conversion.to_original(fresh))
        for name in fresh:
            torch.testing.assert_close(reimported[name], fresh[name], rtol=0, atol=0)

    def test_non_involutive_permutation(self):
        original = torch.arange(10).reshape(5, 2)
        transform = ReorderChunks((0, 3, 1, 4, 2))
        converted = transform.forward((original,))
        assert converted[0].tolist() == [[0, 1], [6, 7], [2, 3], [8, 9], [4, 5]]
        torch.testing.assert_close(transform.inverse(converted)[0], original, rtol=0, atol=0)
        assert not torch.equal(transform.forward(converted)[0], original)

    def test_custom_many_to_many_transform(self):
        class SwapAndTranspose:
            def forward(self, tensors):
                first, second = tensors
                return second.T, first.T

            def inverse(self, tensors):
                second, first = tensors
                return first.T, second.T

        conversion = Conversion(rules=(Rule(("a", "b"), ("c", "d"), SwapAndTranspose()),))
        original = {"a": torch.arange(6).reshape(2, 3), "b": torch.arange(12).reshape(4, 3)}
        converted = conversion.to_diffusers(original)
        torch.testing.assert_close(converted["c"], original["b"].T)
        torch.testing.assert_close(converted["d"], original["a"].T)
        for key, value in conversion.to_original(converted).items():
            torch.testing.assert_close(value, original[key])

    @pytest.mark.parametrize(
        "mapping,rules",
        [
            ({"a": "c", "b": "c"}, ()),
            ({"a": "b"}, (Rule(("a",), ("c",)),)),
            ({"a": "b"}, (Rule(("c",), ("b",)),)),
        ],
    )
    def test_collisions_are_rejected_before_conversion(self, mapping, rules):
        with pytest.raises(ValueError, match="keys must be unique"):
            Conversion(mapping, rules)

    @pytest.mark.parametrize("keys", [("a", "a"), (), "a"])
    def test_invalid_rule_keys(self, keys):
        with pytest.raises(ValueError):
            Rule(keys, ("b",))

    def test_missing_group_member_and_unknown_key(self):
        conversion = Conversion(rules=(Rule(("qkv",), ("q", "k", "v"), Split((2, 2, 2))),))
        with pytest.raises(ValueError, match=r"Diffusers -> original: missing keys \['v'\]"):
            conversion.to_original({"q": torch.zeros(2), "k": torch.zeros(2)})
        with pytest.raises(ValueError, match=r"unexpected keys \['typo'\]"):
            conversion.to_diffusers({"qkv": torch.zeros(6), "typo": torch.zeros(1)})

    def test_shape_error_identifies_rule_and_direction(self):
        conversion = Conversion(rules=(Rule(("qkv",), ("q", "k", "v"), Split((2, 2, 2))),))
        with pytest.raises(ValueError, match=r"original -> Diffusers.*qkv.*Split sizes"):
            conversion.to_diffusers({"qkv": torch.zeros(7)})
        with pytest.raises(ValueError, match=r"Diffusers -> original.*Expected split piece shape"):
            conversion.to_original({"q": torch.zeros(2, 4), "k": torch.zeros(2, 3), "v": torch.zeros(2, 4)})

    def test_wrong_number_of_transform_outputs(self):
        conversion = Conversion(rules=(Rule(("a",), ("b", "c"), Identity()),))
        with pytest.raises(ValueError, match="Expected 2 output tensors, got 1"):
            conversion.to_diffusers({"a": torch.ones(1)})

    @pytest.mark.parametrize("second", [torch.ones(2, dtype=torch.float16), torch.empty(2, device="meta")])
    def test_concat_rejects_implicit_dtype_or_device_changes(self, second):
        with pytest.raises(ValueError, match="same dtype and device"):
            Split((2, 2)).inverse((torch.ones(2), second))

    @pytest.mark.parametrize("sizes", [(), (0, 2), (-1, 2)])
    def test_invalid_split_sizes(self, sizes):
        with pytest.raises(ValueError, match="positive"):
            Split(sizes)

    @pytest.mark.parametrize("order", [(), (0, 0), (1, 2)])
    def test_invalid_permutation(self, order):
        with pytest.raises(ValueError, match="permutation"):
            ReorderChunks(order)

    @pytest.mark.parametrize("shape,dim", [((), 0), ((2, 3), 2), ((2, 3), -3)])
    def test_invalid_dimension(self, shape, dim):
        with pytest.raises(ValueError, match="dimension"):
            Split((1, 1), dim).forward((torch.zeros(shape),))
        with pytest.raises(ValueError, match="dimension"):
            ReorderChunks((1, 0), dim).forward((torch.zeros(shape),))

    def test_reorder_requires_equal_nonempty_chunks(self):
        for size in (0, 5):
            with pytest.raises(ValueError, match="Cannot divide"):
                ReorderChunks((1, 0)).forward((torch.zeros(size),))

    def test_meta_tensor_conversion(self):
        conversion = Conversion(rules=(Rule(("qkv",), ("q", "k", "v"), Split((6, 2, 2))),))
        original = {"qkv": torch.empty(10, 4, device="meta")}
        converted = conversion.to_diffusers(original)
        assert [tensor.shape for tensor in converted.values()] == [(6, 4), (2, 4), (2, 4)]
        assert conversion.to_original(converted)["qkv"].shape == (10, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_gate_has_a_canonical_inverse(dtype, bias):
    from diffusers.loaders.conversion import FoldLinearGate

    operation = FoldLinearGate(bias=bias)
    gate = torch.tensor([0.0, -0.25, 1.75], dtype=dtype)
    weight = torch.randn(3, 5, dtype=dtype)
    original = (gate, weight, torch.randn(3, dtype=dtype)) if bias else (gate, weight)
    copies = tuple(t.clone() for t in original)
    folded = operation.forward(original)
    expected = (gate.float()[:, None] * weight.float()).to(dtype)
    torch.testing.assert_close(folded[0], expected, rtol=0, atol=0)
    canonical = operation.inverse(folded)
    torch.testing.assert_close(canonical[0], torch.ones_like(gate), rtol=0, atol=0)
    for actual, wanted in zip(operation.forward(canonical), folded):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for actual, wanted in zip(original, copies):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    original_keys = ("gate", "weight", "bias") if bias else ("gate", "weight")
    converted_keys = ("weight", "bias") if bias else ("weight",)
    conversion = Conversion(rules=(Rule(original_keys, converted_keys, operation),))
    assert not conversion.lossless


def test_composition_reports_normalization_and_freezes_steps():
    from diffusers.loaders.conversion import Chain, FoldLinearGate, Reverse

    steps = [FoldLinearGate(bias=False)]
    operation = Chain(steps)
    steps.clear()
    assert len(operation.transforms) == 1
    assert not operation.lossless
    assert not Reverse(operation).lossless
    assert Conversion(mapping={"a": "b"}).lossless


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lora_layout_preserves_scaling_and_factorization(dtype):
    from diffusers.loaders.conversion import get_conversion

    module = "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
    config = {"modules": [module], "original_format": "kohya", "include_alpha": True, "use_dora": True}
    stem = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    original = {
        stem + ".lora_down.weight": torch.randn(2, 8, dtype=dtype),
        stem + ".lora_up.weight": torch.randn(8, 2, dtype=dtype),
        stem + ".alpha": torch.tensor(0.5, dtype=torch.float32),
        stem + ".dora_scale": torch.randn(8, dtype=dtype),
    }
    conversion = get_conversion("LoRA", config)
    converted = conversion.to_diffusers(original)
    assert converted[module + ".alpha"].dtype == torch.float32
    assert torch.equal(converted[module + ".lora_A.weight"], original[stem + ".lora_down.weight"])
    for key, tensor in conversion.to_original(converted).items():
        torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)


def test_lora_underscore_collision_is_rejected():
    from diffusers.loaders.conversion import get_conversion

    with pytest.raises(ValueError, match="collision"):
        get_conversion("LoRA", {"modules": ["unet.a_b.c", "unet.a.b_c"]})


def test_flux_ip_adapter_keeps_distinct_projection_and_value_weights():
    from diffusers.loaders.conversion import get_conversion

    conversion = get_conversion("FluxIPAdapter", {"num_layers": 2})
    original = {key: torch.tensor([i], dtype=torch.float32) for i, key in enumerate(sorted(conversion.original_keys))}
    converted = conversion.to_diffusers(original)
    assert torch.equal(converted["image_proj.proj.weight"], original["ip_adapter_proj_model.proj.weight"])
    assert torch.equal(
        converted["ip_adapter.1.to_v_ip.weight"],
        original["double_blocks.1.processor.ip_adapter_double_stream_v_proj.weight"],
    )
    assert not torch.equal(converted["ip_adapter.1.to_v_ip.weight"], converted["ip_adapter.1.to_k_ip.weight"])
    for key, tensor in conversion.to_original(converted).items():
        torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
