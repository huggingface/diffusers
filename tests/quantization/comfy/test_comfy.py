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

import importlib.metadata
import json
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version
from torch import nn
from torch.nn import functional as F

from diffusers import ComfyQuantizationConfig, QwenImage21Transformer2DModel
from diffusers.loaders.single_file_utils import (
    convert_minimax_h3_transformer_checkpoint_to_diffusers,
    convert_qwen_image21_transformer_checkpoint_to_diffusers,
    convert_z_image_transformer_checkpoint_to_diffusers,
)
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.quantizers.comfy import ComfyQuantizer
from diffusers.utils import is_comfy_kitchen_available


if is_comfy_kitchen_available() and Version(importlib.metadata.version("comfy-kitchen")) >= Version("0.2.41"):
    import comfy_kitchen as kitchen
    from comfy_kitchen.float_utils import from_blocked, to_blocked
    from comfy_kitchen.tensor import QuantizedTensor

    from diffusers.quantizers.comfy.comfy_quantizer import COMFY_FORMATS
    from diffusers.quantizers.comfy.utils import ComfyLinear, ComfyQuantizedTensor

else:
    pytest.skip("Requires comfy-kitchen>=0.2.41", allow_module_level=True)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_config_and_registration(dtype):
    config = ComfyQuantizationConfig(compute_dtype=dtype, pre_quantized=False)
    assert config.pre_quantized is True
    assert not hasattr(config, "quant_type")
    quantizer = DiffusersAutoQuantizer.from_config(config.to_dict())
    assert isinstance(quantizer, ComfyQuantizer)
    assert quantizer.compute_dtype == dtype
    assert quantizer.pre_quantized is True
    assert json.loads(config.to_json_string(use_diff=False))["compute_dtype"] == str(dtype).removeprefix("torch.")
    with pytest.raises(ValueError, match="compute_dtype"):
        ComfyQuantizationConfig(compute_dtype=torch.float32)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_layer_metadata_and_tensor_lifecycle(quant_format):
    state = checkpoint(quant_format)
    original = state["linear.weight"].clone()
    q = bundle(state)["linear.weight"]
    assert list(state) == ["linear.weight"]
    assert isinstance(q, QuantizedTensor)
    assert q.shape == (96, 256)
    assert q.dtype == torch.bfloat16
    torch.testing.assert_close(q._qdata, original, rtol=0, atol=0)
    for copy in (q.detach(), q.clone(), q.to(torch.float16), nn.Parameter(q, requires_grad=False)):
        assert isinstance(copy, ComfyQuantizedTensor)
        assert copy.storage_dtype == original.dtype
        torch.testing.assert_close(copy._qdata, original, rtol=0, atol=0)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_split_and_cat_keep_codes_and_sidecars_aligned(quant_format):
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
def test_blocked_scale_conversion_preserves_distinct_rows(quant_format):
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


@pytest.mark.parametrize("quant_format", FORMATS)
@pytest.mark.parametrize(
    "conversion,source,targets",
    [
        (
            convert_z_image_transformer_checkpoint_to_diffusers,
            "layers.0.attention.qkv",
            [f"layers.0.attention.to_{x}" for x in "qkv"],
        ),
        (
            convert_qwen_image21_transformer_checkpoint_to_diffusers,
            "transformer_blocks.0.img_mlp.gate_up",
            ["transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.0.img_mlp.proj"],
        ),
    ],
)
def test_existing_converters_preserve_quantized_weights(quant_format, conversion, source, targets):
    state = bundle(checkpoint(quant_format, source, rows=32 * len(targets)))
    q = state[f"{source}.weight"]
    original_data, original_fields = q._row_components()
    converted = conversion(state)
    for index, target in enumerate(targets):
        part = converted[f"{target}.weight"]
        assert isinstance(part, ComfyQuantizedTensor)
        data, fields = part._row_components()
        torch.testing.assert_close(data.float(), original_data[index * 32 : (index + 1) * 32].float(), rtol=0, atol=0)
        for name, (value, axis) in fields.items():
            torch.testing.assert_close(
                value.float(), original_fields[name][0].narrow(axis, index * 32, 32).float(), rtol=0, atol=0
            )


@pytest.mark.parametrize("layout", ["stacked", "interleaved"])
def test_existing_h3_converter_row_reordering(layout):
    state = bundle(checkpoint(prefix="blocks.0.attn.qkv_proj", rows=96))
    q = state["blocks.0.attn.qkv_proj.weight"]
    converted = convert_minimax_h3_transformer_checkpoint_to_diffusers(
        state, config={"num_attention_heads": 4, "attention_head_dim": 8}, qkv_layout=layout
    )
    order = torch.arange(96)
    if layout == "interleaved":
        order = order.reshape(4, 3, 8).transpose(0, 1).flatten()
    for index, name in enumerate("qkv"):
        part = converted[f"transformer_blocks.0.attn.to_{name}.weight"]
        assert isinstance(part, ComfyQuantizedTensor)
        rows = order[index * 32 : (index + 1) * 32]
        torch.testing.assert_close(part._qdata, q._qdata[rows], rtol=0, atol=0)
        torch.testing.assert_close(part.params.scale, q.params.scale[rows], rtol=0, atol=0)


def test_cat_rejects_different_shared_scales_and_metadata():
    first = bundle(checkpoint("float8_e4m3fn"))["linear.weight"]
    second = first.clone()
    second._params = type(first.params)(scale=torch.tensor(0.5), orig_dtype=first.dtype, orig_shape=tuple(first.shape))
    with pytest.raises(ValueError, match="different scale"):
        torch.cat([first, second])
    with pytest.raises(NotImplementedError, match="input features"):
        first.chunk(2, dim=1)


@pytest.mark.parametrize("metadata", [{"format": "unknown"}, {"convrot": True}])
def test_invalid_format_is_rejected(metadata):
    state = checkpoint()
    state["linear.comfy_quant"] = marker(**metadata)
    with pytest.raises(ValueError, match="supported Comfy format"):
        bundle(state)


def test_missing_scale_and_marker_are_rejected():
    state = checkpoint()
    state.pop("linear.weight_scale")
    with pytest.raises(ValueError, match="requires weight_scale"):
        bundle(state)
    with pytest.raises(ValueError, match="no native Comfy"):
        bundle({"linear.weight": torch.zeros(32, 256, dtype=torch.int8)})


def test_missing_optional_dependency():
    with patch("diffusers.quantizers.comfy.comfy_quantizer.is_comfy_kitchen_available", return_value=False):
        with pytest.raises(ImportError, match="comfy-kitchen"):
            ComfyQuantizer(ComfyQuantizationConfig()).validate_environment()


def test_mixed_formats_load_through_public_api(tmp_path):
    model = QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=64,
        num_attention_heads=4,
        context_in_dim=8,
        mlp_ratio=1,
        axes_dims_rope=(16, 24, 24),
    ).to(torch.bfloat16)
    model.save_config(tmp_path)
    state = model.state_dict()
    state.update(checkpoint(prefix="transformer_blocks.0.attn.to_q", rows=256))
    state.update(
        checkpoint("float8_e4m3fn", prefix="transformer_blocks.0.attn.to_k", rows=256, full_precision_matrix_mult=True)
    )
    fused = checkpoint("nvfp4", prefix="transformer_blocks.0.img_mlp.gate_up", rows=512)
    state.pop("transformer_blocks.0.img_mlp.gate_layer.weight")
    state.pop("transformer_blocks.0.img_mlp.proj.weight")
    state.update(fused)
    loaded = QwenImage21Transformer2DModel.from_single_file(
        state, config=str(tmp_path), quantization_config=ComfyQuantizationConfig()
    )
    assert isinstance(loaded.transformer_blocks[0].attn.to_q, ComfyLinear)
    assert loaded.transformer_blocks[0].attn.to_q.weight.storage_dtype == torch.int8
    assert loaded.transformer_blocks[0].attn.to_k.weight.storage_dtype == torch.float8_e4m3fn
    assert loaded.transformer_blocks[0].img_mlp.proj.weight.storage_dtype == torch.uint8
    torch.testing.assert_close(loaded.img_in.weight, model.img_in.weight)
    assert all(not value.is_meta for value in loaded.state_dict().values())


def test_fp32_exclusion_dequantizes_inside_quantizer():
    model = nn.Sequential(nn.Linear(256, 96, bias=False, device="meta"))
    state = bundle(checkpoint(prefix="0"))
    expected = state["0.weight"].to(torch.float32).dequantize()
    quantizer = ComfyQuantizer(ComfyQuantizationConfig())
    quantizer.preprocess_model(model, device_map=None, state_dict=state, keep_in_fp32_modules=["0"])
    quantizer.create_quantized_param(model, state["0.weight"].to(torch.float32), "0.weight", "cpu", state)
    assert type(model[0]) is nn.Linear
    assert model[0].weight.dtype == torch.float32
    torch.testing.assert_close(model[0].weight, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rotated", [False, True])
def test_int8_runtime_against_independent_reference(dtype, rotated):
    torch.manual_seed(42)
    state = checkpoint(rows=32, convrot=rotated, convrot_groupsize=64)
    q = bundle(state)["linear.weight"].to(device="cuda", dtype=dtype)
    linear = ComfyLinear(256, 32, bias=True, device="cuda", dtype=dtype)
    linear.weight = nn.Parameter(q, requires_grad=False)
    x = torch.randn(2, 7, 512, device="cuda", dtype=dtype)[..., ::2]
    rotated_x = x.float()
    if rotated:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device="cuda", dtype=torch.float32
        )
        h = torch.kron(torch.kron(h4, h4), h4) / 8
        rotated_x = (rotated_x.reshape(2, 7, -1, 64) @ h).reshape(2, 7, 256)
    expected = F.linear(rotated_x, q._qdata.float() * q.params.scale, linear.bias.float())
    with torch.inference_mode(), kitchen.use_backend("cuda"):
        actual = linear(x)
    assert actual.dtype == dtype
    assert (actual.float() - expected).norm() / expected.norm() < 0.025


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cpu_offload_and_dtype_move_keep_sidecars():
    from accelerate import cpu_offload_with_hook

    q = bundle(checkpoint(rows=32, convrot=True, convrot_groupsize=64))["linear.weight"]
    q.sidecars["input_scale"] = torch.tensor(0.1234567)
    scale = q.params.scale.clone()
    linear = ComfyLinear(256, 32, bias=False, dtype=torch.bfloat16)
    linear.weight = nn.Parameter(q, requires_grad=False)
    model, hook = cpu_offload_with_hook(nn.Sequential(linear), execution_device="cuda:0")
    inputs = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        first = model(inputs)
        hook.offload()
        assert linear.weight.device.type == "cpu"
        assert linear.weight.sidecars["input_scale"].device.type == "cpu"
        second = model(inputs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    model.to(dtype=torch.float16)
    assert linear.weight.dtype == torch.float16
    assert linear.weight.params.scale.dtype == torch.float32
    torch.testing.assert_close(linear.weight.params.scale.cpu(), scale, rtol=0, atol=0)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_linear_runtime_matches_layout_reference(quant_format):
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


def test_svdquant_tile_packed_qkv_conversion():
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
        torch.testing.assert_close(part.params.proj_up, expected_up[index * 128 : (index + 1) * 128], rtol=0, atol=0)


def test_full_precision_mm_keeps_input_scaling():
    state = checkpoint("float8_e4m3fn", full_precision_matrix_mult=True)
    state["linear.pre_quant_scale"] = torch.full((256,), 0.5)
    q = bundle(state)["linear.weight"]
    linear = ComfyLinear(256, 96, bias=False, dtype=torch.bfloat16)
    linear.weight = nn.Parameter(q, requires_grad=False)
    inputs = torch.randn(2, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(linear(inputs), F.linear(inputs * 0.5, q.dequantize()))


@pytest.mark.parametrize(
    "suffix,value,message",
    [
        ("weight_scale", torch.ones(2), "per-output-row shape"),
        ("weight_scale", torch.ones(96, 1, dtype=torch.float16), "FP32"),
        ("weight_scale", torch.full((96, 1), float("nan")), "finite positive"),
        ("weight_scale", torch.zeros(96, 1), "finite positive"),
        ("weight", torch.ones(96, 256), "shape or dtype"),
        ("comfy_quant", torch.tensor([255], dtype=torch.uint8), "valid JSON"),
        ("comfy_quant", torch.tensor(1), "one-dimensional uint8"),
    ],
)
def test_malformed_checkpoint_is_rejected(suffix, value, message):
    state = checkpoint()
    state[f"linear.{suffix}"] = value
    with pytest.raises(ValueError, match=message):
        bundle(state)


@pytest.mark.parametrize("group_size", [True, 0, 2, 8, 63, 1024])
def test_invalid_rotation_group_is_rejected(group_size):
    with pytest.raises(ValueError, match="power of four|group-aligned"):
        bundle(checkpoint(convrot=True, convrot_groupsize=group_size))


def test_unmarked_fp8_weight_is_rejected():
    state = checkpoint()
    state["other.weight"] = torch.ones(32, 256).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="markers"):
        bundle(state)
