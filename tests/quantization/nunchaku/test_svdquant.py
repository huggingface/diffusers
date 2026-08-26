# coding=utf-8
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

"""CPU-only tests for data-free Nunchaku SVDQuant quantization.

These tests intentionally avoid importing ``diffusers.quantizers.nunchaku.utils``
(which requires the ``kernels`` package and a CUDA GPU); the packed layouts are
validated against pure-torch reference unpackers ported from DeepCompressor.
"""

import pytest
import torch

from diffusers import NunchakuLiteQuantizationConfig
from diffusers.quantizers.nunchaku.svdquant import (
    _NunchakuWeightPacker,
    pack_data_free_bias,
    quantize_linear_data_free,
)


# ---------------------------------------------------------------------------
# Reference unpackers (ported from DeepCompressor's Nunchaku converter).
# ---------------------------------------------------------------------------


def _ceil_divide(x, divisor):
    return (x + divisor - 1) // divisor


def _unpack_weight(packed, rows, columns):
    p = _NunchakuWeightPacker()
    padded_rows = _ceil_divide(rows, p.mem_n) * p.mem_n
    padded_columns = _ceil_divide(columns, p.mem_k * p.num_k_unrolls) * p.mem_k * p.num_k_unrolls
    unpacked = packed.contiguous().view(torch.int32)
    unpacked = unpacked.view(
        padded_rows // p.mem_n,
        padded_columns // p.mem_k,
        p.num_k_packs,
        p.num_n_packs,
        p.num_n_lanes,
        p.num_k_lanes,
        p.n_pack_size,
        p.k_pack_size,
        p.reg_n,
    )
    shift = torch.arange(0, 32, 4, dtype=torch.int32)
    unpacked = unpacked.unsqueeze(-1).bitwise_right_shift(shift).bitwise_and(0xF)
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    unpacked = unpacked.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    return unpacked.view(padded_rows, padded_columns)[:rows, :columns]


def _unpack_vector(packed, rows):
    p = _NunchakuWeightPacker()
    padded_rows = _ceil_divide(rows, p.warp_n) * p.warp_n
    s_pack_size = min(max(p.warp_n // p.num_lanes, 2), 8)
    num_s_lanes = min(p.num_lanes, p.warp_n // s_pack_size)
    num_s_packs = p.warp_n // (s_pack_size * num_s_lanes)
    unpacked = packed.contiguous().view(
        padded_rows // p.warp_n, 1, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2
    )
    unpacked = unpacked.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return unpacked.view(padded_rows)[:rows]


def _unpack_group_scale(packed, rows, groups):
    p = _NunchakuWeightPacker()
    padded_rows = _ceil_divide(rows, p.warp_n) * p.warp_n
    padded_groups = _ceil_divide(groups, p.num_k_unrolls) * p.num_k_unrolls
    s_pack_size = min(max(p.warp_n // p.num_lanes, 2), 8)
    num_s_lanes = min(p.num_lanes, p.warp_n // s_pack_size)
    num_s_packs = p.warp_n // (s_pack_size * num_s_lanes)
    unpacked = packed.contiguous().view(
        padded_rows // p.warp_n, padded_groups, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2
    )
    unpacked = unpacked.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return unpacked.view(padded_rows, padded_groups)[:rows, :groups]


def _unpack_micro_scale(packed, rows, groups):
    p = _NunchakuWeightPacker()
    padded_rows = _ceil_divide(rows, p.warp_n) * p.warp_n
    group_fragment = p.insn_k // 16
    padded_groups = _ceil_divide(groups, group_fragment) * group_fragment
    s_pack_size = min(max(p.warp_n // p.num_lanes, 1), 4)
    num_s_packs = _ceil_divide(p.warp_n, s_pack_size * 32)
    unpacked = packed.contiguous().view(
        padded_rows // p.warp_n, padded_groups // group_fragment, num_s_packs, 8, 4, s_pack_size, group_fragment
    )
    unpacked = unpacked.permute(0, 2, 5, 4, 3, 1, 6).contiguous()
    return unpacked.view(padded_rows, padded_groups)[:rows, :groups]


def _unpack_lowrank(packed, down, rows, columns):
    p = _NunchakuWeightPacker()
    reg_n, reg_k = 1, 2
    pack_n = p.n_pack_size * p.num_n_lanes * reg_n
    pack_k = p.k_pack_size * p.num_k_lanes * reg_k
    padded_rows = _ceil_divide(rows, pack_n) * pack_n
    padded_columns = _ceil_divide(columns, pack_k) * pack_k
    if down:
        r, c = padded_rows, padded_columns
        r_packs, c_packs = r // pack_n, c // pack_k
    else:
        c, r = padded_rows, padded_columns
        c_packs, r_packs = c // pack_n, r // pack_k
    unpacked = packed.contiguous().view(
        c_packs, r_packs, p.num_n_lanes, p.num_k_lanes, p.n_pack_size, p.k_pack_size, reg_n, reg_k
    )
    unpacked = unpacked.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
    unpacked = unpacked.view(c_packs, r_packs, pack_n, pack_k)
    if down:
        unpacked = unpacked.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        unpacked = unpacked.permute(0, 2, 1, 3).contiguous().view(c, r)
    return unpacked[:rows, :columns]


def _fp4_codebook():
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _reconstruct(state, out_features, in_features, group_size, rank, precision):
    """Rebuild the original weight from the packed data-free state."""

    codes = _unpack_weight(state["qweight"], out_features, in_features).float()
    groups = in_features // group_size
    if precision == "nvfp4":
        values = _fp4_codebook()[codes.long() & 0xF]
        wscales = _unpack_micro_scale(state["wscales"].view(torch.float8_e4m3fn), out_features, groups).float()
        scale = wscales * state["wtscale"].float()
    else:
        values = codes
        wscales = _unpack_group_scale(state["wscales"], out_features, groups).float()
        scale = wscales
    residual = values.view(out_features, groups, group_size) * scale.view(out_features, groups, 1)
    residual = residual.view(out_features, in_features)
    smooth = _unpack_vector(state["smooth_factor"], in_features).float()
    down = _unpack_lowrank(state["proj_down"], down=True, rows=rank, columns=in_features).float()
    up = _unpack_lowrank(state["proj_up"], down=False, rows=out_features, columns=rank).float()
    # Residual is in smoothed coordinates; the low-rank branch already absorbed 1/smooth.
    return residual / smooth.view(1, -1) + up @ down


OUT_FEATURES, IN_FEATURES = 256, 384


@pytest.mark.parametrize("precision,group_size", [("int4", 64), ("nvfp4", 16)])
def test_data_free_state_shapes_and_dtypes(precision, group_size):
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    state = quantize_linear_data_free(weight, precision=precision, group_size=group_size, rank=32)

    assert state["qweight"].shape == (OUT_FEATURES, IN_FEATURES // 2)
    assert state["qweight"].dtype == torch.int8
    assert state["smooth_factor"].shape == (IN_FEATURES,)
    assert state["proj_down"].shape == (IN_FEATURES, 32)
    assert state["proj_up"].shape == (OUT_FEATURES, 32)
    assert state["wscales"].shape == (IN_FEATURES // group_size, OUT_FEATURES)
    if precision == "nvfp4":
        assert state["wscales"].dtype == torch.float8_e4m3fn
        assert state["wcscales"].shape == (OUT_FEATURES,)
        assert torch.all(state["wcscales"].float() == 1.0)
        assert state["wtscale"].shape == (1,)
    else:
        assert state["wscales"].dtype == torch.bfloat16
        assert "wcscales" not in state
        assert "wtscale" not in state
    for tensor in (state["smooth_factor"], state["proj_down"], state["proj_up"]):
        assert tensor.dtype == torch.bfloat16


@pytest.mark.parametrize("precision,group_size", [("int4", 64), ("nvfp4", 16)])
def test_data_free_round_trip_error_bounded(precision, group_size):
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)

    state = quantize_linear_data_free(weight, precision=precision, group_size=group_size, rank=32)
    reconstructed = _reconstruct(state, OUT_FEATURES, IN_FEATURES, group_size, 32, precision)
    error = (reconstructed - weight).norm() / weight.norm()

    state_rank0 = quantize_linear_data_free(weight, precision=precision, group_size=group_size, rank=0)
    reconstructed_rank0 = _reconstruct(state_rank0, OUT_FEATURES, IN_FEATURES, group_size, 0, precision)
    error_rank0 = (reconstructed_rank0 - weight).norm() / weight.norm()

    assert error < 0.15
    assert error < error_rank0


def test_data_free_bias_round_trip():
    torch.manual_seed(0)
    bias = torch.randn(OUT_FEATURES)
    packed = pack_data_free_bias(bias)
    assert packed.shape == (OUT_FEATURES,)
    assert packed.dtype == torch.bfloat16
    assert torch.allclose(_unpack_vector(packed, OUT_FEATURES).float(), bias, atol=1e-2, rtol=1e-2)


def test_data_free_rejects_unsupported_dimensions():
    with pytest.raises(ValueError, match="multiples of 128"):
        quantize_linear_data_free(torch.randn(100, 384), precision="int4", group_size=64, rank=32)
    with pytest.raises(ValueError, match="multiple of 16"):
        quantize_linear_data_free(torch.randn(256, 384), precision="int4", group_size=64, rank=24)
    with pytest.raises(ValueError, match="Unsupported precision"):
        quantize_linear_data_free(torch.randn(256, 384), precision="fp8", group_size=64, rank=32)


def test_config_accepts_pre_quantized_flag():
    config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32, "targets": ["proj"]}
    )
    assert config.pre_quantized is True

    config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32, "targets": ["proj"]},
        pre_quantized=False,
    )
    assert config.pre_quantized is False


def test_config_rejects_data_free_awq():
    with pytest.raises(NotImplementedError, match="svdq_w4a4"):
        NunchakuLiteQuantizationConfig(
            awq_w4a16={"precision": "int4", "group_size": 64, "targets": ["proj"]},
            pre_quantized=False,
        )


def test_quantizer_create_quantized_param_fills_module():
    from diffusers.quantizers.nunchaku.nunchaku_quantizer import NunchakuLiteQuantizer

    config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32, "targets": ["proj"]},
        pre_quantized=False,
    )
    quantizer = NunchakuLiteQuantizer(config, pre_quantized=False)
    assert quantizer.pre_quantized is False

    class StubQuantizedLinear(torch.nn.Module):
        precision = "nvfp4"
        group_size = 16
        rank = 32

    class StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = StubQuantizedLinear()

    model = StubModel()
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    quantizer.create_quantized_param(model, weight, "proj.weight", torch.device("cpu"))
    quantizer.create_quantized_param(model, torch.randn(OUT_FEATURES), "proj.bias", torch.device("cpu"))

    parameters = dict(model.proj.named_parameters())
    for name in ("qweight", "wscales", "wcscales", "wtscale", "smooth_factor", "proj_down", "proj_up", "bias"):
        assert name in parameters, f"missing quantized parameter {name}"
        assert not parameters[name].requires_grad
    assert parameters["qweight"].shape == (OUT_FEATURES, IN_FEATURES // 2)
    assert parameters["bias"].shape == (OUT_FEATURES,)


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.norm_linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)  # adaLN-style
        self.frozen = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.odd_shape = torch.nn.Linear(100, OUT_FEATURES)


class _InferenceToyModel(torch.nn.Module):
    _keep_in_fp32_modules = ["frozen"]

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_ToyBlock() for _ in range(2)])
        self.embedder = torch.nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.proj_out = torch.nn.Linear(OUT_FEATURES, IN_FEATURES)


def test_infer_data_free_targets():
    from diffusers.quantizers.nunchaku.svdquant import infer_data_free_targets

    model = _InferenceToyModel()
    # Default: restricted to the repeated `blocks` stack (embedder/proj_out are
    # outside), minus adaLN-style names ("norm"), _keep_in_fp32_modules, and
    # dimension-ineligible layers.
    targets = infer_data_free_targets(model, group_size=16)
    assert targets == ["blocks.0.proj", "blocks.1.proj"]

    # An explicit list replaces the default name patterns.
    targets = infer_data_free_targets(model, group_size=16, exclude_targets=[])
    assert sorted(targets) == ["blocks.0.norm_linear", "blocks.0.proj", "blocks.1.norm_linear", "blocks.1.proj"]

    targets = infer_data_free_targets(model, group_size=16, exclude_targets=["blocks.0", "norm"])
    assert targets == ["blocks.1.proj"]

    with pytest.raises(ValueError, match="Could not infer"):
        infer_data_free_targets(model, group_size=16, exclude_targets=["proj", "norm"])


def test_quantizer_infers_targets_when_omitted(monkeypatch):
    import sys
    import types

    from diffusers.quantizers.nunchaku.nunchaku_quantizer import NunchakuLiteQuantizer

    config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32},
        pre_quantized=False,
        exclude_targets=["norm_linear"],
    )
    quantizer = NunchakuLiteQuantizer(config, pre_quantized=False)
    model = _InferenceToyModel()

    # Stub out `.utils` (its import fetches the CUDA kernels) so only the
    # target-inference part of _process_model_before_weight_loading runs.
    stub = types.ModuleType("diffusers.quantizers.nunchaku.utils")
    stub.replace_with_nunchaku_linear = lambda target_model, quantization_config, compute_dtype: len(
        quantization_config["svdq_w4a4"]["targets"]
    )
    stub.check_strict_state_dict_match = None
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.nunchaku.utils", stub)

    quantizer._process_model_before_weight_loading(model)

    assert config.svdq_w4a4["targets"] == ["blocks.0.proj", "blocks.1.proj"]


def test_config_targets_optional_only_for_data_free():
    config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32},
        pre_quantized=False,
    )
    assert config.svdq_w4a4.get("targets") is None

    with pytest.raises(ValueError, match="missing required field 'targets'"):
        NunchakuLiteQuantizationConfig(svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32})


def test_quantizer_update_missing_keys_filters_data_free_params():
    from diffusers.quantizers.nunchaku.nunchaku_quantizer import NunchakuLiteQuantizer

    config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32},
        pre_quantized=False,
    )
    quantizer = NunchakuLiteQuantizer(config, pre_quantized=False)
    missing = ["blocks.0.proj.qweight", "blocks.0.proj.smooth_factor", "blocks.0.proj.wtscale", "other.weight"]
    assert quantizer.update_missing_keys(None, missing, prefix="") == ["other.weight"]

    pre_config = NunchakuLiteQuantizationConfig(
        svdq_w4a4={"precision": "nvfp4", "group_size": 16, "rank": 32, "targets": ["blocks.0.proj"]}
    )
    quantizer_pre = NunchakuLiteQuantizer(pre_config, pre_quantized=True)
    assert quantizer_pre.pre_quantized is True
    assert quantizer_pre.update_missing_keys(None, missing, prefix="") == missing
