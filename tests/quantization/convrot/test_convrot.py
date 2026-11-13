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

from diffusers import (
    AutoencoderKLMiniMaxH3,
    ComfyQuantizationConfig,
    GGUFQuantizationConfig,
    MiniMaxH3Transformer3DModel,
    QwenImage21Transformer2DModel,
)
from diffusers.hooks.tensor_parallel import resolve_tp_shard_specs
from diffusers.loaders.single_file_utils import (
    convert_minimax_h3_transformer_checkpoint_to_diffusers,
    convert_minimax_h3_vae_checkpoint_to_diffusers,
    convert_qwen_image21_transformer_checkpoint_to_diffusers,
    convert_z_image_transformer_checkpoint_to_diffusers,
    infer_minimax_h3_checkpoint_config,
    infer_minimax_h3_vae_checkpoint_config,
)
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.quantizers.comfy import ComfyQuantizer
from diffusers.quantizers.comfy.utils import ComfyInt8Linear


@pytest.fixture
def kitchen():
    ck = pytest.importorskip("comfy_kitchen")
    if Version(importlib.metadata.version("comfy-kitchen")) < Version("0.2.41"):
        pytest.skip("Requires comfy-kitchen>=0.2.41")
    return ck


@pytest.fixture
def native_kitchen(kitchen):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("Requires an NVIDIA GPU with INT8 tensor cores")
    backend = kitchen.list_backends().get("cuda", {})
    if not backend.get("available") or "int8_linear" not in backend.get("capabilities", []):
        pytest.skip("Requires Comfy-Kitchen's native CUDA INT8 backend")
    return kitchen


def marker(**kwargs):
    payload = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256, **kwargs}
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


def checkpoint(prefix="linear", rows=32, columns=256, **kwargs):
    return {
        f"{prefix}.weight": torch.randint(-127, 128, (rows, columns), dtype=torch.int8),
        f"{prefix}.weight_scale": torch.linspace(0.001, 0.004, rows).reshape(rows, 1),
        f"{prefix}.comfy_quant": marker(**kwargs),
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_config_and_registration(dtype):
    config = ComfyQuantizationConfig(compute_dtype=dtype)
    assert config.compute_dtype == dtype
    assert isinstance(DiffusersAutoQuantizer.from_config(config), ComfyQuantizer)
    restored = DiffusersAutoQuantizer.from_dict(config.to_dict())
    assert isinstance(restored, ComfyQuantizationConfig)
    assert restored.compute_dtype == dtype
    assert isinstance(DiffusersAutoQuantizer.from_config(config.to_dict()), ComfyQuantizer)
    with pytest.raises(ValueError, match="compute_dtype"):
        ComfyQuantizationConfig(torch.float32)
    with pytest.raises(ValueError, match="must match"):
        ComfyQuantizer(config).update_torch_dtype(torch.float32)
    with pytest.raises(ValueError, match="prequantized"):
        ComfyQuantizationConfig(pre_quantized=False)
    assert json.loads(config.to_json_string(use_diff=False))["compute_dtype"] == str(dtype).replace("torch.", "")
    assert json.loads(config.to_json_string())["quant_method"] == "comfy"
    assert ("compute_dtype" in config.to_diff_dict()) == (dtype != torch.bfloat16)
    assert config.quant_type == "int8_convrot"
    assert restored.quant_type == "int8_convrot"
    assert ComfyQuantizationConfig(quant_type="int8_convrot").quant_type == config.quant_type


@pytest.mark.parametrize("quant_type", ["nvfp4", "mxfp8", "int4", "int8_tensorwise"])
def test_unsupported_quantization_type(quant_type):
    with pytest.raises(ValueError, match="quant_type"):
        ComfyQuantizationConfig(quant_type=quant_type)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"quant_type": "nvfp4"}, "quant_type"),
        ({"compute_dtype": torch.float32}, "compute_dtype"),
        ({"pre_quantized": False}, "prequantized"),
    ],
)
def test_config_overrides_are_validated_before_loading(overrides, message):
    config = ComfyQuantizationConfig.from_dict(ComfyQuantizationConfig().to_dict(), **overrides)
    with pytest.raises(ValueError, match=message):
        DiffusersAutoQuantizer.from_config(config)


def test_string_compute_dtype_override_is_normalized_before_loading():
    config = ComfyQuantizationConfig.from_dict(ComfyQuantizationConfig().to_dict(), compute_dtype="float16")
    quantizer = DiffusersAutoQuantizer.from_config(config)
    assert quantizer.compute_dtype == torch.float16


@pytest.mark.parametrize("rotation", [False, None])
def test_convrot_quantization_type_rejects_checkpoint_without_rotation(rotation):
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(256, 32, bias=False))
    state = checkpoint(convrot=False)
    if rotation is None:
        payload = {"format": "int8_tensorwise"}
        state["linear.comfy_quant"] = torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)
    with pytest.raises(ValueError, match="requires a checkpoint with ConvRot"):
        ComfyQuantizer(ComfyQuantizationConfig()).preprocess_model(model, device_map=None, state_dict=state)
    assert isinstance(model.linear, nn.Linear)
    assert "linear.comfy_quant" in state


def test_convrot_checkpoint_can_include_unrotated_layers():
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(256, 32, bias=False))
    model.add_module("unrotated", nn.Linear(256, 32, bias=False))
    state = {**checkpoint(), **checkpoint("unrotated", convrot=False)}
    ComfyQuantizer(ComfyQuantizationConfig()).preprocess_model(model, device_map=None, state_dict=state)
    assert isinstance(model.linear, ComfyInt8Linear)
    assert model.linear.convrot
    assert isinstance(model.unrotated, ComfyInt8Linear)
    assert not model.unrotated.convrot


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
def test_fused_weight_scale_marker_conversion(conversion, source, targets):
    state = checkpoint(source, rows=32 * len(targets))
    weight, scale, metadata = (
        state[f"{source}.{suffix}"].clone() for suffix in ("weight", "weight_scale", "comfy_quant")
    )
    converted = conversion(state)
    assert not any(source + "." in name for name in converted)
    for index, target in enumerate(targets):
        torch.testing.assert_close(converted[f"{target}.weight"], weight[index * 32 : (index + 1) * 32])
        torch.testing.assert_close(converted[f"{target}.weight_scale"], scale[index * 32 : (index + 1) * 32])
        torch.testing.assert_close(converted[f"{target}.comfy_quant"], metadata)


def test_z_image_output_projection_metadata_is_renamed():
    converted = convert_z_image_transformer_checkpoint_to_diffusers(
        checkpoint("model.diffusion_model.layers.0.attention.out")
    )
    assert set(converted) == {
        f"layers.0.attention.to_out.0.{suffix}" for suffix in ("weight", "weight_scale", "comfy_quant")
    }


@pytest.mark.parametrize(
    "conversion,source,parts",
    [
        (convert_z_image_transformer_checkpoint_to_diffusers, "layers.0.attention.qkv", 3),
        (convert_qwen_image21_transformer_checkpoint_to_diffusers, "transformer_blocks.0.img_mlp.gate_up", 2),
    ],
)
def test_bf16_conversion_is_unchanged(conversion, source, parts):
    weight = torch.randn(parts * 16, 32, dtype=torch.bfloat16)
    result = conversion({f"{source}.weight": weight})
    torch.testing.assert_close(torch.cat(list(result.values()), dim=0), weight)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda state: state.pop("linear.weight_scale"), "requires both"),
        (lambda state: state.update({"linear.comfy_quant": marker(format="nvfp4")}), "int8_tensorwise"),
        (lambda state: state.update({"linear.comfy_quant": marker(format="mxfp8")}), "int8_tensorwise"),
        (lambda state: state.update({"linear.comfy_quant": marker(format="float8_e4m3fn")}), "int8_tensorwise"),
        (lambda state: state.update({"linear.comfy_quant": marker(format="svdquant_w4a4")}), "int8_tensorwise"),
        (lambda state: state.update({"linear.comfy_quant": marker(convrot_groupsize=128)}), "power of four"),
        (lambda state: state.update({"linear.comfy_quant": marker(convrot_groupsize=1024)}), "group-aligned"),
        (
            lambda state: state.update({"linear.comfy_quant": marker(full_precision_matrix_mult=True)}),
            "full_precision",
        ),
        (lambda state: state.update({"linear.weight_scale": torch.zeros(32, 1)}), "finite positive"),
        (lambda state: state.update({"linear.weight_scale": torch.full((32, 1), float("nan"))}), "finite positive"),
        (lambda state: state.update({"linear.weight_scale": torch.ones(7)}), "per-output-row"),
        (lambda state: state.update({"linear.weight": state["linear.weight"].float()}), "int8 tensor"),
        (lambda state: state.update({"linear.comfy_quant": torch.tensor([255], dtype=torch.uint8)}), "valid JSON"),
    ],
)
def test_invalid_checkpoint_is_rejected_before_replacement(change, message):
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(256, 32, bias=False))
    state = checkpoint()
    change(state)
    quantizer = ComfyQuantizer(ComfyQuantizationConfig())
    with pytest.raises(ValueError, match=message):
        quantizer.preprocess_model(model, device_map=None, state_dict=state)
    assert isinstance(model.linear, nn.Linear)


def test_missing_optional_dependency():
    with patch("diffusers.quantizers.comfy.comfy_quantizer.is_comfy_kitchen_available", return_value=False):
        with pytest.raises(ImportError, match="comfy-kitchen"):
            ComfyQuantizer(ComfyQuantizationConfig()).validate_environment()


def test_unmarked_int8_weight_is_rejected():
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(256, 32, bias=False))
    state = checkpoint()
    state["unmarked.weight"] = torch.zeros(32, 256, dtype=torch.int8)
    with pytest.raises(ValueError, match="require native Comfy quantization markers"):
        ComfyQuantizer(ComfyQuantizationConfig()).preprocess_model(model, device_map=None, state_dict=state)


def test_bad_fused_scale_rows_are_rejected():
    state = checkpoint("layers.0.attention.qkv", rows=96)
    state["layers.0.attention.qkv.weight_scale"] = torch.ones(32, 1)
    with pytest.raises(ValueError, match="one entry per fused output row"):
        convert_z_image_transformer_checkpoint_to_diffusers(state)


def test_single_file_load_preserves_codes_scales_and_unquantized_weights(tmp_path, kitchen):
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
    quantized = checkpoint("transformer_blocks.0.attn.to_q", rows=256)
    state.update(quantized)
    fused_prefix = "transformer_blocks.0.img_mlp.gate_up"
    fused = checkpoint(fused_prefix, rows=512)
    state.pop("transformer_blocks.0.img_mlp.gate_layer.weight")
    state.pop("transformer_blocks.0.img_mlp.proj.weight")
    state.update(fused)
    loaded = QwenImage21Transformer2DModel.from_single_file(
        state,
        config=str(tmp_path),
        quantization_config=ComfyQuantizationConfig(),
        dtype=torch.bfloat16,
    )
    linear = loaded.transformer_blocks[0].attn.to_q
    assert isinstance(linear, ComfyInt8Linear)
    torch.testing.assert_close(linear.weight, quantized["transformer_blocks.0.attn.to_q.weight"])
    torch.testing.assert_close(
        linear.weight_scale, quantized["transformer_blocks.0.attn.to_q.weight_scale"], rtol=0, atol=0
    )
    torch.testing.assert_close(loaded.img_in.weight, model.img_in.weight)
    for index, name in enumerate(("gate_layer", "proj")):
        projection = getattr(loaded.transformer_blocks[0].img_mlp, name)
        assert isinstance(projection, ComfyInt8Linear)
        torch.testing.assert_close(projection.weight, fused[f"{fused_prefix}.weight"][index * 256 : (index + 1) * 256])
        torch.testing.assert_close(
            projection.weight_scale,
            fused[f"{fused_prefix}.weight_scale"][index * 256 : (index + 1) * 256],
            rtol=0,
            atol=0,
        )
    original_scale = linear.weight_scale.clone()
    linear.to(dtype=torch.float16)
    assert linear.weight.dtype == torch.int8
    assert linear.weight_scale.dtype == torch.float32
    torch.testing.assert_close(linear.weight_scale, original_scale, rtol=0, atol=0)
    assert all(value.device.type != "meta" for value in loaded.state_dict().values())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rotated", [True, False])
@pytest.mark.parametrize("group_size", [64, 256])
def test_native_int8_linear_against_independent_reference(dtype, rotated, group_size, native_kitchen):
    ck = native_kitchen
    torch.manual_seed(42)
    linear = ComfyInt8Linear(
        nn.Linear(256, 32, bias=True), (32, 1), {"convrot": rotated, "convrot_groupsize": group_size}, dtype
    )
    linear.weight = nn.Parameter(
        torch.randint(-64, 65, (32, 256), device="cuda", dtype=torch.int8), requires_grad=False
    )
    linear.weight_scale = torch.full((32, 1), 1 / 256, device="cuda", dtype=torch.float32)
    linear.bias = nn.Parameter(torch.randn(32, device="cuda", dtype=dtype), requires_grad=False)
    x = torch.randn(2, 7, 512, device="cuda", dtype=dtype)[..., ::2]
    rotated_x = x.float()
    if rotated:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device="cuda", dtype=torch.float32
        )
        h = h4
        for _ in range((group_size.bit_length() - 1) // 2 - 1):
            h = torch.kron(h, h4)
        rotated_x = (rotated_x.reshape(2, 7, -1, group_size) @ (h / group_size**0.5)).reshape(2, 7, 256)
    reference = torch.nn.functional.linear(rotated_x, linear.weight.float() * linear.weight_scale, linear.bias.float())
    with ck.use_backend("cuda"):
        output = linear(x)
    assert output.shape == (2, 7, 32)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    relative_error = (output.float() - reference).norm() / reference.norm()
    assert relative_error < 0.025, relative_error.item()


def test_repeated_forward_and_model_cpu_offload(native_kitchen):
    from accelerate import cpu_offload_with_hook

    ck = native_kitchen
    linear = ComfyInt8Linear(nn.Linear(256, 32, bias=False), (32, 1), {"convrot": True}, torch.bfloat16)
    linear.weight = nn.Parameter(torch.randint(-32, 33, (32, 256), dtype=torch.int8), requires_grad=False)
    linear.weight_scale = torch.full((32, 1), 0.00314159, dtype=torch.float32)
    scale = linear.weight_scale.clone()
    model = nn.Sequential(linear, nn.LayerNorm(32, dtype=torch.bfloat16))
    model, hook = cpu_offload_with_hook(model, execution_device=torch.device("cuda:0"))
    inputs = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), ck.use_backend("cuda"):
        first = model(inputs)
        second = model(inputs)
        hook.offload()
        assert linear.weight.device.type == "cpu"
        assert linear.weight_scale.device.type == "cpu"
        third = model(inputs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, third, rtol=0, atol=0)
    torch.testing.assert_close(linear.weight_scale.cpu(), scale, rtol=0, atol=0)
    assert linear.weight.dtype == torch.int8
    assert linear.weight_scale.dtype == torch.float32


@pytest.mark.parametrize("layout", ["stacked", "interleaved"])
@pytest.mark.parametrize("scalar_scale", [False, True])
def test_h3_fused_qkv_and_swiglu_metadata(layout, scalar_scale):
    source = "blocks.0.attn.qkv_proj"
    state = checkpoint(source, rows=12, columns=4, convrot_groupsize=4)
    state[f"{source}.weight"] = torch.arange(48, dtype=torch.int8).reshape(12, 4)
    state.update(checkpoint("blocks.0.mlp.fc1", rows=16, columns=4, convrot_groupsize=4))
    for projection, rows in ((source, 12), ("blocks.0.mlp.fc1", 16)):
        state[f"{projection}.bias"] = torch.arange(rows, dtype=torch.float32)
        if scalar_scale:
            state[f"{projection}.weight_scale"] = torch.tensor(0.001)
            state[f"{projection}.comfy_quant"] = marker(convrot=False)
    weight = state[f"{source}.weight"].clone()
    bias = state[f"{source}.bias"].clone()
    scale = state[f"{source}.weight_scale"].clone()
    ff_weight = state["blocks.0.mlp.fc1.weight"].clone()
    ff_scale = state["blocks.0.mlp.fc1.weight_scale"].clone()
    ff_bias = state["blocks.0.mlp.fc1.bias"].clone()
    converted = convert_minimax_h3_transformer_checkpoint_to_diffusers(
        state, config={"num_attention_heads": 2, "attention_head_dim": 2}, qkv_layout=layout
    )
    row_order = torch.arange(12)
    if layout == "interleaved":
        row_order = row_order.view(2, 3, 2).permute(1, 0, 2).reshape(-1)
    for index, name in enumerate("qkv"):
        target = f"transformer_blocks.0.attn.to_{name}"
        indices = row_order[index * 4 : (index + 1) * 4]
        torch.testing.assert_close(converted[f"{target}.weight"], weight[indices])
        torch.testing.assert_close(converted[f"{target}.bias"], bias[indices], rtol=0, atol=0)
        torch.testing.assert_close(
            converted[f"{target}.weight_scale"], scale if scalar_scale else scale[indices], rtol=0, atol=0
        )
        assert json.loads(bytes(converted[f"{target}.comfy_quant"].tolist()))["convrot"] is (not scalar_scale)
    indices = torch.cat([torch.arange(8, 16), torch.arange(8)])
    torch.testing.assert_close(converted["transformer_blocks.0.ff.net.0.proj.weight"], ff_weight[indices])
    torch.testing.assert_close(
        converted["transformer_blocks.0.ff.net.0.proj.weight_scale"],
        ff_scale if scalar_scale else ff_scale[indices],
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(converted["transformer_blocks.0.ff.net.0.proj.bias"], ff_bias[indices], rtol=0, atol=0)


def h3_config():
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 24,
        "num_layers": 1,
        "num_refiner_layers": 1,
        "ffn_dim": 32,
        "in_channels": 4,
        "audio_in_channels": 6,
        "text_dim": 8,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 8,
        "rope_freq_dim": 2,
    }


def h3_inputs():
    return {
        "hidden_states": torch.randn(1, 4, 16),
        "audio_hidden_states": torch.randn(1, 2, 6),
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "timestep": torch.tensor([0.0, 0.5, 1.0]),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2]),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 0, 0, 0, 0]),
        "position_ids": torch.arange(9).float()[:, None].expand(-1, 3),
        "text_indices": torch.arange(3),
        "audio_indices": torch.arange(3, 5),
        "video_indices": torch.arange(5, 9),
    }


def test_h3_pruned_curve_matches_full_timestep_path_on_grid():
    torch.manual_seed(11)
    full = MiniMaxH3Transformer3DModel(**h3_config()).eval()
    pruned = MiniMaxH3Transformer3DModel(**h3_config(), adaln_curve_grid=5).eval()
    state = {
        key.replace("adaln_proj.linear.", "adaln_proj.curve_proj.").replace(
            "norm_out.linear.", "norm_out.curve_proj."
        ): value
        for key, value in full.state_dict().items()
        if not key.startswith("time_embedder.")
    }
    with torch.inference_mode():
        grid = torch.linspace(0, 1, 5)
        state["adaln_t_table"] = torch.nn.functional.silu(full.time_embedder(full.time_proj(grid)))
        pruned.load_state_dict(state, strict=True)
        inputs = h3_inputs()
        expected = full(**inputs)
        actual = pruned(**inputs)
    torch.testing.assert_close(actual.sample, expected.sample, rtol=0, atol=1e-5)
    torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=0, atol=1e-5)
    assert not hasattr(pruned, "time_embedder")


def test_h3_pruned_curve_interpolation_and_endpoint_clamping():
    model = MiniMaxH3Transformer3DModel(**h3_config(), adaln_curve_grid=5).eval()
    with torch.no_grad():
        model.adaln_t_table.copy_(torch.arange(40).reshape(5, 8))
    seen = []
    model.transformer_blocks[0].adaln_proj.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    inputs = h3_inputs()
    inputs["timestep"] = torch.tensor([-0.5, 0.375, 1.5])
    with torch.inference_mode():
        output = model(**inputs)
    expected = torch.stack(
        [model.adaln_t_table[0], (model.adaln_t_table[1] + model.adaln_t_table[2]) / 2, model.adaln_t_table[-1]]
    )
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    assert torch.isfinite(output.sample).all()
    assert torch.isfinite(output.audio_sample).all()


def test_h3_curve_config_inference_and_validation():
    assert infer_minimax_h3_checkpoint_config({}) == {}
    assert infer_minimax_h3_checkpoint_config({"adaln_t_table": torch.zeros(1025, 8)}) == {
        "adaln_curve_grid": 1025,
        "time_embed_dim": 8,
    }
    with pytest.raises(ValueError, match="adaln_t_table"):
        infer_minimax_h3_checkpoint_config({"adaln_t_table": torch.zeros(1, 8)})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("low_cpu_mem_usage", [True, False])
def test_h3_pruned_single_file_infers_architecture_and_dtype(tmp_path, dtype, low_cpu_mem_usage):
    config = h3_config()
    model = MiniMaxH3Transformer3DModel(**config, adaln_curve_grid=5).eval()
    with torch.no_grad():
        model.adaln_t_table.normal_()
    state = model.state_dict()
    source = {
        key.replace("transformer_blocks.", "blocks.")
        .replace(".adaln_proj.curve_proj.", ".adaln_proj.linear.")
        .replace("norm_out.curve_proj.", "final_layer.adaln_proj.linear."): value
        for key, value in state.items()
    }
    MiniMaxH3Transformer3DModel(**config).save_config(tmp_path)
    loaded = MiniMaxH3Transformer3DModel.from_single_file(
        source, config=str(tmp_path), dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage
    )
    assert loaded.config.adaln_curve_grid == 5
    assert loaded.config.time_embed_dim == 8
    assert loaded.adaln_t_table.dtype == dtype
    assert loaded.transformer_blocks[0].adaln_proj.curve_proj.weight.dtype == dtype
    assert loaded.norm_out.curve_proj.weight.dtype == dtype
    assert loaded.transformer_blocks[0].attn.to_q.weight.dtype == dtype
    torch.testing.assert_close(loaded.adaln_t_table, model.adaln_t_table.to(dtype), rtol=0, atol=0)
    assert not any(value.is_meta for value in loaded.state_dict().values())
    with torch.inference_mode():
        output = loaded(**h3_inputs())
    assert torch.isfinite(output.sample).all()
    assert torch.isfinite(output.audio_sample).all()


@pytest.mark.parametrize(
    "adaln_curve_grid",
    [
        None,
        pytest.param(
            5,
            marks=pytest.mark.xfail(
                raises=AttributeError,
                strict=True,
                reason="Pruned AdaLN uses curve_proj; dynamic TP plans await diffusers#15197.",
            ),
        ),
    ],
)
def test_h3_tensor_parallel_plan_resolves_modulation_projection(adaln_curve_grid):
    model = MiniMaxH3Transformer3DModel(**h3_config(), adaln_curve_grid=adaln_curve_grid)
    specs = resolve_tp_shard_specs(model, model._tp_plan, tp_degree=2)
    projection = "linear" if adaln_curve_grid is None else "curve_proj"
    prefix = f"transformer_blocks.0.adaln_proj.{projection}"
    assert specs[f"{prefix}.weight"].dim == 1
    assert specs[f"{prefix}.weight"].block_sizes == [model.config.time_embed_dim]
    assert specs[f"{prefix}.bias"].dim is None
    assert "transformer_blocks.*.adaln_proj.linear" in MiniMaxH3Transformer3DModel._tp_plan


def test_h3_pruned_from_pretrained_round_trip(tmp_path):
    torch.manual_seed(11)
    model = MiniMaxH3Transformer3DModel(**h3_config(), adaln_curve_grid=5).eval()
    with torch.no_grad():
        model.adaln_t_table.normal_()
    model.save_pretrained(tmp_path)
    loaded = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path)
    assert loaded.config.adaln_curve_grid == 5
    inputs = h3_inputs()
    with torch.inference_mode():
        expected = model(**inputs)
        actual = loaded(**inputs)
    torch.testing.assert_close(actual.sample, expected.sample, rtol=0, atol=0)
    torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=0, atol=0)
    loaded = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert loaded.adaln_t_table.dtype == torch.float32
    assert loaded.transformer_blocks[0].adaln_proj.curve_proj.weight.dtype == torch.float32
    assert loaded.norm_out.curve_proj.weight.dtype == torch.float32
    assert loaded.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16


def test_h3_quantized_adaln_preserves_fractional_activations(native_kitchen):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation

    torch.manual_seed(42)
    modulation = MiniMaxH3AdaLayerNormModulation(time_embed_dim=64, hidden_size=32)
    projection = ComfyInt8Linear(modulation.linear, (576, 1), {"convrot": False}, torch.bfloat16)
    projection.weight = nn.Parameter(
        torch.randint(-64, 65, (576, 64), device="cuda", dtype=torch.int8), requires_grad=False
    )
    projection.weight_scale = torch.full((576, 1), 1 / 256, device="cuda", dtype=torch.float32)
    projection.bias = nn.Parameter(torch.randn(576, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    modulation.linear = projection
    temb = torch.randn(3, 64, device="cuda") * 0.5
    reference = torch.nn.functional.linear(
        torch.nn.functional.silu(temb), projection.weight.float() * projection.weight_scale, projection.bias.float()
    ).reshape(-1, 192)
    with torch.inference_mode(), native_kitchen.use_backend("cuda"):
        actual = torch.cat(modulation(temb), dim=-1)
    assert actual.dtype == torch.bfloat16
    assert (actual.float() - reference).norm() / reference.norm() < 0.025


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32])
@pytest.mark.parametrize("scalar_scale", [False, True])
def test_h3_vae_fused_conversion(dtype, scalar_scale):
    prefix = "decoder.transformer_blocks.0"
    state = {}
    for projection, rows in (("attn.to_qkv", 96), ("ff.w1", 64)):
        source = f"{prefix}.{projection}"
        state.update(checkpoint(source, rows=rows))
        state[f"{source}.weight"] = state[f"{source}.weight"].to(dtype)
        state[f"{source}.bias"] = torch.arange(rows, dtype=torch.float32)
        if scalar_scale:
            state[f"{source}.weight_scale"] = torch.tensor(0.001)
    config = {"decoder_num_attention_heads": 2, "decoder_attention_head_dim": 16}
    converted = convert_minimax_h3_vae_checkpoint_to_diffusers(state, config=config)
    for index, name in enumerate("qkv"):
        indices = torch.cat(
            (torch.arange(index * 16, (index + 1) * 16), torch.arange(48 + index * 16, 48 + (index + 1) * 16))
        )
        for suffix in ("weight", "bias", "weight_scale", "comfy_quant"):
            original = state[f"{prefix}.attn.to_qkv.{suffix}"]
            expected = original if suffix == "comfy_quant" or original.ndim == 0 else original[indices]
            torch.testing.assert_close(converted[f"{prefix}.attn.to_{name}.{suffix}"], expected, rtol=0, atol=0)
    for suffix in ("weight", "bias", "weight_scale", "comfy_quant"):
        original = state[f"{prefix}.ff.w1.{suffix}"]
        expected = (
            original if suffix == "comfy_quant" or original.ndim == 0 else torch.cat((original[32:], original[:32]))
        )
        torch.testing.assert_close(converted[f"{prefix}.ff.net.0.proj.{suffix}"], expected, rtol=0, atol=0)
    twice = convert_minimax_h3_vae_checkpoint_to_diffusers(converted, config={})
    assert twice.keys() == converted.keys()
    for key in converted:
        torch.testing.assert_close(twice[key], converted[key], rtol=0, atol=0)


@pytest.mark.parametrize("projection,rows", [("attn.to_qkv", 95), ("ff.w1", 63)])
def test_h3_vae_invalid_fused_rows(projection, rows):
    with pytest.raises(ValueError, match="interleaved QKV output rows|even number"):
        convert_minimax_h3_vae_checkpoint_to_diffusers(
            checkpoint(f"decoder.transformer_blocks.0.{projection}", rows=rows),
            config={"decoder_num_attention_heads": 2, "decoder_attention_head_dim": 16},
        )


@pytest.mark.parametrize("projection", ["attn.to_qkv", "ff.w1"])
@pytest.mark.parametrize("suffix", ["weight_scale", "bias"])
def test_h3_vae_invalid_fused_sidecar(projection, suffix):
    state = checkpoint(f"decoder.transformer_blocks.0.{projection}", rows=96)
    state[f"decoder.transformer_blocks.0.{projection}.{suffix}"] = torch.ones(32)
    with pytest.raises(ValueError, match="one entry per fused output row"):
        convert_minimax_h3_vae_checkpoint_to_diffusers(
            state, config={"decoder_num_attention_heads": 2, "decoder_attention_head_dim": 16}
        )


@pytest.mark.parametrize(
    "name,value", [("latents_mean", [float("nan")]), ("latents_std", [0.0]), ("latents_std", [[1.0]])]
)
def test_h3_vae_invalid_normalization(name, value):
    with pytest.raises(ValueError, match="finite per-channel"):
        infer_minimax_h3_vae_checkpoint_config({name: torch.tensor(value)})


def test_h3_vae_normalization_channel_mismatch():
    with pytest.raises(ValueError, match="24 channels"):
        convert_minimax_h3_vae_checkpoint_to_diffusers({"latents_mean": torch.ones(4)}, config={"latent_channels": 24})


@pytest.fixture
def h3_vae_checkpoint(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(42)
    model = AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        norm_num_groups=8,
        decoder_num_layers=1,
        decoder_num_attention_heads=4,
        decoder_attention_head_dim=64,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=1,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )
    model.save_config(tmp_path / "vae")
    state = model.state_dict()
    prefix = "decoder.transformer_blocks.0"
    for suffix in ("weight", "bias"):
        state[f"{prefix}.attn.to_qkv.{suffix}"] = torch.stack(
            [state.pop(f"{prefix}.attn.to_{name}.{suffix}").unflatten(0, (4, 64)) for name in "qkv"], dim=1
        ).flatten(0, 2)
    for source, rows in (("attn.to_qkv", 768), ("attn.to_out.0", 256), ("ff.net.0.proj", 512), ("ff.net.2", 256)):
        state.update(checkpoint(f"{prefix}.{source}", rows=rows))
    original = {}
    for key, value in state.items():
        key = key.replace("encoder.down_blocks.", "encoder.down.")
        key = key.replace(".resnets.", ".block.").replace(".downsamplers.0.", ".downsample.")
        key = key.replace(".conv_shortcut.", ".nin_shortcut.").replace("decoder.proj_in.", "decoder.x_embedder.")
        key = key.replace(".attn.to_out.0.", ".attn.to_out.")
        key = key.replace(".ff.net.0.proj.", ".ff.w1.").replace(".ff.net.2.", ".ff.w2.")
        original[key] = value.contiguous()
    original.update(
        latents_mean=torch.arange(4).float(),
        latents_std=torch.arange(1, 5).float(),
        **{"decoder.mask_token": torch.zeros(1, 1, 256)},
    )
    path = tmp_path / "vae.safetensors"
    save_file(original, path)
    return path, original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_h3_vae_single_file_load(tmp_path, h3_vae_checkpoint, kitchen, dtype):
    path, original = h3_vae_checkpoint
    class_fp32_modules = list(AutoencoderKLMiniMaxH3._keep_in_fp32_modules)
    model = AutoencoderKLMiniMaxH3.from_single_file(
        str(path),
        config=str(tmp_path),
        subfolder="vae",
        quantization_config=ComfyQuantizationConfig(compute_dtype=dtype),
        dtype=dtype,
    )
    assert model.config.latents_mean == [0.0, 1.0, 2.0, 3.0]
    assert model.config.latents_std == [1.0, 2.0, 3.0, 4.0]
    assert model.use_tiling
    assert AutoencoderKLMiniMaxH3._keep_in_fp32_modules == class_fp32_modules
    assert model._keep_in_fp32_modules is not AutoencoderKLMiniMaxH3._keep_in_fp32_modules
    other_model = AutoencoderKLMiniMaxH3.from_config(model.config)
    assert other_model._keep_in_fp32_modules == class_fp32_modules
    assert model.dtype == dtype
    assert sum(isinstance(module, ComfyInt8Linear) for module in model.modules()) == 6
    assert not any(value.is_meta for value in model.state_dict().values())
    converted = convert_minimax_h3_vae_checkpoint_to_diffusers(original, config=model.config)
    for name, value in model.state_dict().items():
        if name.endswith(".weight_scale") or any(part in name.split(".") for part in model._keep_in_fp32_modules):
            expected_dtype = torch.float32
        elif name.endswith(".weight") and converted[name].dtype == torch.int8:
            expected_dtype = torch.int8
        else:
            expected_dtype = dtype
        assert value.dtype == expected_dtype, name
        torch.testing.assert_close(value, converted[name].to(expected_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("include_markers", [True, False])
def test_h3_vae_requires_quantization_config(tmp_path, h3_vae_checkpoint, include_markers):
    _, state = h3_vae_checkpoint
    if not include_markers:
        state = {key: value for key, value in state.items() if not key.endswith(".comfy_quant")}
    with pytest.raises(ValueError, match="quantization_config"):
        AutoencoderKLMiniMaxH3.from_single_file(state, config=str(tmp_path), subfolder="vae")


def test_single_file_floating_weights_with_other_quantization_metadata(tmp_path, h3_vae_checkpoint):
    _, state = h3_vae_checkpoint
    state = {
        key: marker(format="float8_e4m3fn") if key.endswith(".comfy_quant") else value.float()
        for key, value in state.items()
        if not key.endswith(".weight_scale")
    }
    loaded = AutoencoderKLMiniMaxH3.from_single_file(state, config=str(tmp_path), subfolder="vae")
    assert not getattr(loaded, "is_quantized", False)
    assert not any(value.is_meta for value in loaded.state_dict().values())
    torch.testing.assert_close(loaded.decoder.proj_in.weight, state["decoder.x_embedder.weight"])


def test_comfy_checkpoint_rejects_other_quantization_backend(tmp_path, h3_vae_checkpoint):
    _, state = h3_vae_checkpoint
    with pytest.raises(ValueError, match="ComfyQuantizationConfig"):
        AutoencoderKLMiniMaxH3.from_single_file(
            state, config=str(tmp_path), subfolder="vae", quantization_config=GGUFQuantizationConfig()
        )


def test_h3_vae_missing_tensor_is_rejected(tmp_path, h3_vae_checkpoint, kitchen):
    _, state = h3_vae_checkpoint
    state.pop("decoder.transformer_blocks.0.attn.to_qkv.bias")
    with pytest.raises(ValueError, match="missing model tensors"):
        AutoencoderKLMiniMaxH3.from_single_file(
            state, config=str(tmp_path), subfolder="vae", quantization_config=ComfyQuantizationConfig()
        )


def test_h3_vae_native_encode_decode(tmp_path, h3_vae_checkpoint, native_kitchen):
    path, _ = h3_vae_checkpoint
    model = AutoencoderKLMiniMaxH3.from_single_file(
        str(path), config=str(tmp_path), subfolder="vae", quantization_config=ComfyQuantizationConfig()
    ).to("cuda")
    with torch.inference_mode(), native_kitchen.use_backend("cuda"):
        encoded = model.encode(torch.zeros(1, 3, 1, 8, 8, device="cuda")).latent_dist.mean
        decoded = model.decode(torch.zeros(1, 4, 7, 2, 2, device="cuda")).sample
    assert encoded.shape == (1, 4, 1, 2, 2)
    assert decoded.shape == (1, 3, 22, 8, 8)
    assert torch.isfinite(encoded).all() and torch.isfinite(decoded).all()
