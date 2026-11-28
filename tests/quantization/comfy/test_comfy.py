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

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusers import ComfyQuantizationConfig
from diffusers.quantizers.comfy import ComfyQuantizer
from diffusers.quantizers.comfy.comfy_quantizer import COMFY_FORMATS
from diffusers.utils import is_comfy_kitchen_available

from ...testing_utils import is_comfy, is_quantization, require_comfy_kitchen


if is_comfy_kitchen_available():
    import comfy_kitchen as kitchen
    from comfy_kitchen.float_utils import from_blocked, to_blocked
    from comfy_kitchen.tensor import QuantizedTensor

    from diffusers.quantizers.comfy.utils import ComfyLinear, ComfyQuantizedTensor


FORMATS = list(COMFY_FORMATS)


def marker(**metadata):
    return torch.tensor(list(json.dumps(metadata).encode()), dtype=torch.uint8)


def checkpoint(quant_format="int8_tensorwise", prefix="linear", rows=96, columns=256, **metadata):
    torch.manual_seed(10)
    layout, dtype, bits = COMFY_FORMATS[quant_format]
    weight = torch.randint(0, 64, (rows, columns * bits // 8), dtype=torch.uint8).to(dtype)
    state = {"weight": weight}
    if quant_format == "nvfp4":
        state.update(
            weight_scale=to_blocked(torch.ones(rows, columns // 16).to(torch.float8_e4m3fn)),
            weight_scale_2=torch.tensor(0.01),
        )
    elif quant_format == "mxfp8":
        state["weight_scale"] = to_blocked(torch.full((rows, columns // 32), 127, dtype=torch.uint8))
    elif quant_format in ("asym_w4a8_int8", "w6a8_int8"):
        state.update(
            weight_s_rel=torch.ones(rows, columns // 16).to(torch.float8_e4m3fn),
            weight_s_channel=torch.full((rows,), 0.01),
        )
    elif quant_format == "convrot_w4a4":
        state["weight_scale"] = torch.full((rows,), 0.01)
    elif quant_format in ("awq_w4a16", "svdquant_w4a4"):
        state["weight_scale"] = torch.full((columns // 64, rows), 0.01)
        if quant_format == "awq_w4a16":
            state["weight_zero"] = torch.randn(columns // 64, rows) * 0.001
        elif quant_format == "svdquant_w4a4":
            state.update(
                proj_down=torch.randn(columns, 4) * 0.01,
                proj_up=torch.randn(rows, 4) * 0.01,
                smooth_factor=torch.ones(columns),
            )
    else:
        state["weight_scale"] = (
            torch.linspace(0.001, 0.004, rows).reshape(rows, 1)
            if quant_format == "int8_tensorwise"
            else torch.tensor(0.01)
        )
    state["comfy_quant"] = marker(format=quant_format, **metadata)
    return {f"{prefix}.{name}": value for name, value in state.items()}


def bundle(state):
    return ComfyQuantizer(ComfyQuantizationConfig()).maybe_update_state_dict(state)


def padded_checkpoint(quant_format, rows, columns, prefix="linear"):
    torch.manual_seed(0)
    layout = COMFY_FORMATS[quant_format][0]
    q = QuantizedTensor.from_float(torch.randn(rows, columns), layout)
    state = {"weight": q._qdata, "comfy_quant": marker(format=quant_format)}
    if quant_format == "nvfp4":
        state.update(weight_scale=q.params.block_scale, weight_scale_2=q.params.scale)
    else:
        state["weight_scale"] = q.params.scale
    return {f"{prefix}.{name}": value for name, value in state.items()}, q


@is_quantization
@is_comfy
@require_comfy_kitchen
class TestComfyQuantization:
    @pytest.mark.parametrize("quant_format", FORMATS)
    def test_split_and_cat_keep_codes_and_sidecars_aligned(self, quant_format):
        q = bundle(checkpoint(quant_format))["linear.weight"]
        original_data, original_fields = q._row_components()
        parts = q.split([16, 48, 32], dim=0)
        offset = 0
        for part, rows in zip(parts, (16, 48, 32)):
            assert isinstance(part, ComfyQuantizedTensor)
            assert part.shape == (rows, 256)
            data, fields = part._row_components()
            torch.testing.assert_close(data, original_data[offset : offset + rows], rtol=0, atol=0)
            for name, (value, axis) in fields.items():
                expected = original_fields[name][0].narrow(axis, offset, rows)
                torch.testing.assert_close(value.float(), expected.float(), rtol=0, atol=0)
            offset += rows
        joined = torch.cat(parts[::-1], dim=0)
        data, fields = joined._row_components()
        expected_order = torch.cat([torch.arange(64, 96), torch.arange(16, 64), torch.arange(16)])
        torch.testing.assert_close(data.float(), original_data.float()[expected_order], rtol=0, atol=0)
        for name, (value, axis) in fields.items():
            torch.testing.assert_close(
                value.float(), original_fields[name][0].float().index_select(axis, expected_order), rtol=0, atol=0
            )

    @pytest.mark.parametrize("quant_format", ["nvfp4", "mxfp8"])
    def test_blocked_scale_conversion_preserves_distinct_rows(self, quant_format):
        state = checkpoint(quant_format)
        block = 16 if quant_format == "nvfp4" else 32
        scales = torch.arange(96 * (256 // block)).reshape(96, -1).remainder(20).add(1)
        stored = scales.to(torch.float8_e4m3fn) if block == 16 else scales.add(110).to(torch.uint8)
        state["linear.weight_scale"] = to_blocked(stored)
        q = bundle(state)["linear.weight"]
        parts = q.chunk(3)
        for index, part in enumerate(parts):
            scale = part.params.block_scale if block == 16 else part.params.scale
            decoded = from_blocked(scale, 32, 256 // block)
            torch.testing.assert_close(
                decoded.view(torch.uint8), stored[index * 32 : (index + 1) * 32].view(torch.uint8), rtol=0, atol=0
            )

    def test_negative_transpose_dims_match_positive_dims(self):
        original = bundle(checkpoint(rows=96))["linear.weight"]
        q = original.view(2, 3, 16, 256)
        expected = q.transpose(1, 0)
        actual = q.transpose(-3, 0)
        order = torch.arange(96).reshape(2, 3, 16).transpose(0, 1).flatten()
        assert actual.shape == expected.shape == (3, 2, 16, 256)
        torch.testing.assert_close(actual._qdata, expected._qdata, rtol=0, atol=0)
        torch.testing.assert_close(actual.params.scale, expected.params.scale, rtol=0, atol=0)
        torch.testing.assert_close(actual._qdata, original._qdata[order], rtol=0, atol=0)
        torch.testing.assert_close(actual.params.scale, original.params.scale[order], rtol=0, atol=0)

    @pytest.mark.parametrize("quant_format", ["nvfp4", "mxfp8"])
    def test_padded_weight_restores_linear_shape(self, quant_format):
        state, reference = padded_checkpoint(quant_format, 100, 50, "0")
        state = bundle(state)
        model = nn.Sequential(nn.Linear(50, 100, bias=False, device="meta"))
        quantizer = ComfyQuantizer(ComfyQuantizationConfig())
        quantizer.preprocess_model(model, device_map=None, state_dict=state, keep_in_fp32_modules=[])
        quantizer.create_quantized_param(model, state["0.weight"], "0.weight", "cpu", state)
        assert model[0].weight.shape == (100, 50)
        torch.testing.assert_close(
            model[0].weight.dequantize().float(), reference.dequantize().float(), rtol=0.01, atol=1e-3
        )
        inputs = torch.randn(3, 50, dtype=torch.bfloat16)
        with torch.inference_mode(), kitchen.use_backend("eager"):
            assert model(inputs).shape == (3, 100)

    def test_svdquant_tile_packed_qkv_conversion(self):
        from comfy_kitchen.backends.eager.svdquant import _tile_packed_weight_to_row_major

        rows, columns = 384, 256
        state = checkpoint("svdquant_w4a4", rows=rows)
        expected = state["linear.weight"].clone()
        expected_scale = state["linear.weight_scale"].clone()
        expected_up = state["linear.proj_up"].clone()
        state["linear.weight"] = (
            expected.reshape(3, 32, 4, columns // 64, 32).permute(0, 3, 1, 2, 4).reshape(3, columns // 64, 32, 128)
        )
        state["linear.weight_scale"] = expected_scale.reshape(columns // 64, 3, 128).permute(1, 0, 2)
        state["linear.proj_up"] = expected_up.reshape(3, 128, 4).transpose(1, 2)
        torch.testing.assert_close(_tile_packed_weight_to_row_major(state["linear.weight"]), expected)
        q = bundle(state)["linear.weight"]
        for index, part in enumerate(q.chunk(3)):
            torch.testing.assert_close(part._qdata, expected[index * 128 : (index + 1) * 128], rtol=0, atol=0)
            torch.testing.assert_close(
                part.params.scale, expected_scale[:, index * 128 : (index + 1) * 128], rtol=0, atol=0
            )
            torch.testing.assert_close(
                part.params.proj_up, expected_up[index * 128 : (index + 1) * 128], rtol=0, atol=0
            )

    def test_cat_rejects_different_shared_scales_and_metadata(self):
        first = bundle(checkpoint("float8_e4m3fn"))["linear.weight"]
        second = first.clone()
        second._params = type(first.params)(
            scale=torch.tensor(0.5), orig_dtype=first.dtype, orig_shape=tuple(first.shape)
        )
        with pytest.raises(ValueError, match="different scale"):
            torch.cat([first, second])
        with pytest.raises(NotImplementedError, match="input features"):
            first.chunk(2, dim=1)

    @pytest.mark.parametrize("quant_format", FORMATS)
    def test_linear_runtime_matches_layout_reference(self, quant_format):
        q = bundle(checkpoint(quant_format, rows=128))["linear.weight"]
        linear = ComfyLinear(256, 128, bias=False, dtype=torch.bfloat16)
        linear.weight = nn.Parameter(q, requires_grad=False)
        torch.manual_seed(42)
        inputs = torch.randn(2, 3, 256, dtype=torch.bfloat16)
        with torch.inference_mode(), kitchen.use_backend("eager"):
            actual = linear(inputs)
            reference_input = inputs.reshape(-1, 256)
            if q._layout_cls in ("TensorCoreFP8Layout", "TensorCoreNVFP4Layout", "TensorCoreMXFP8Layout"):
                kwargs = {"dtype": q.storage_dtype} if q._layout_cls == "TensorCoreFP8Layout" else {}
                reference_input = QuantizedTensor.from_float(reference_input, q._layout_cls, **kwargs).dequantize()
            if q._layout_cls == "TensorCoreSVDQuantW4A4Layout":
                reference = F.linear(inputs, QuantizedTensor(q._qdata, q._layout_cls, q.params))
            else:
                reference = F.linear(reference_input, q.dequantize().to(inputs.dtype)).reshape(2, 3, 128)
        assert torch.isfinite(actual).all()
        assert actual.dtype == torch.bfloat16
        relative_error = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert relative_error < (0.25 if quant_format == "convrot_w4a4" else 0.04), relative_error.item()

    def test_full_precision_mm_keeps_input_scaling(self):
        state = checkpoint("float8_e4m3fn", full_precision_matrix_mult=True)
        state["linear.pre_quant_scale"] = torch.full((256,), 0.5)
        q = bundle(state)["linear.weight"]
        linear = ComfyLinear(256, 96, bias=False, dtype=torch.bfloat16)
        linear.weight = nn.Parameter(q, requires_grad=False)
        inputs = torch.randn(2, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(linear(inputs), F.linear(inputs * 0.5, q.dequantize()))
