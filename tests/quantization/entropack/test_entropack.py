import json

import pytest
import torch
import torch.nn as nn

from diffusers import EntroPackConfig, Transformer2DModel
from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.utils import is_entropack_available

from ...testing_utils import require_accelerate, require_torch, require_torch_accelerator


pytestmark = [
    pytest.mark.skipif(not is_entropack_available(), reason="test requires entropack"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA"),
]


def is_entropack_linear(module):
    import entropack

    return isinstance(module, entropack.CompressedLinear)


def tiny_model() -> Transformer2DModel:
    return Transformer2DModel(
        in_channels=8,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        num_layers=1,
        norm_num_groups=2,
        sample_size=8,
    ).to(torch.bfloat16)


def model_inputs(device):
    return (
        torch.randn(1, 8, 8, 8, dtype=torch.bfloat16, device=device),
        torch.randn(1, 4, 16, dtype=torch.bfloat16, device=device),
    )


class TestEntroPackConfig:
    def test_round_trip(self):
        config = EntroPackConfig(
            target_bpp=3.5,
            linear_type="int8",
            modules_to_not_convert=["to_q"],
            entropack_lossy_config={"row_rdo_iterations": 1},
            use_convrot=False,
            convrot_groupsize=64,
        )
        serialized = json.loads(json.dumps(config.to_dict()))
        restored = DiffusersAutoQuantizer.from_dict(serialized)
        assert isinstance(restored, EntroPackConfig)
        assert restored.to_dict() == config.to_dict()

    def test_invalid_values_are_rejected(self):
        for kwargs in (
            {"target_bpp": 100.0},
            {"linear_type": "fp4"},
            {"linear_type": "fp8", "target_bpp": 8.0},
            {"linear_type": "int8", "convrot_groupsize": 8},
            {"linear_type": "int8", "convrot_groupsize": 3},
        ):
            with pytest.raises(ValueError):
                EntroPackConfig(**kwargs)


@require_torch
@require_accelerate
@require_torch_accelerator
class TestEntroPackQuantizer:
    @pytest.mark.parametrize("linear_type", ["default", "int8", "fp8"])
    def test_online_quantize_save_reload(self, tmp_path, linear_type):
        if linear_type == "fp8":
            pytest.importorskip("entropack")
            import entropack

            if torch.cuda.get_device_capability() < entropack.CompressedFP8Linear.min_capability:
                pytest.skip("FP8 requires CUDA capability >= 8.9")
            expected_cls = entropack.CompressedFP8Linear
        elif linear_type == "int8":
            import entropack

            if torch.cuda.get_device_capability() < entropack.CompressedINT8Linear.min_capability:
                pytest.skip("INT8 requires CUDA capability >= 8.0")
            expected_cls = entropack.CompressedINT8Linear
        else:
            import entropack

            expected_cls = entropack.CompressedLinear

        torch.manual_seed(0)
        model = tiny_model()
        model.save_pretrained(tmp_path / "base")
        quantized = Transformer2DModel.from_pretrained(
            tmp_path / "base",
            quantization_config=EntroPackConfig(target_bpp=4.0, linear_type=linear_type),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        layers = [(name, module) for name, module in quantized.named_modules() if is_entropack_linear(module)]
        assert layers
        assert all(isinstance(module, expected_cls) for _, module in layers)
        if linear_type != "default":
            assert all(module.weight_quant_scale.dtype == torch.float32 for _, module in layers)
        if linear_type == "int8":
            assert all(module.use_convrot for _, module in layers)

        x, encoder_hidden_states = model_inputs("cuda")
        with torch.no_grad():
            output = quantized(x, encoder_hidden_states=encoder_hidden_states).sample
        assert torch.isfinite(output).all()

        quantized.save_pretrained(tmp_path / "quantized", max_shard_size="1KB")
        reloaded = Transformer2DModel.from_pretrained(
            tmp_path / "quantized",
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        assert sum(is_entropack_linear(module) for module in reloaded.modules()) == len(layers)
        with torch.no_grad():
            reloaded_output = reloaded(x, encoder_hidden_states=encoder_hidden_states).sample
        torch.testing.assert_close(output, reloaded_output, rtol=0, atol=0)

    def test_int8_convrot_off_round_trip(self, tmp_path):
        model = tiny_model()
        model.save_pretrained(tmp_path / "base")
        quantized = Transformer2DModel.from_pretrained(
            tmp_path / "base",
            quantization_config=EntroPackConfig(target_bpp=4.0, linear_type="int8", use_convrot=False),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        layers = [m for m in quantized.modules() if is_entropack_linear(m)]
        assert layers and not any(m.use_convrot for m in layers)

        x, encoder_hidden_states = model_inputs("cuda")
        with torch.no_grad():
            output = quantized(x, encoder_hidden_states=encoder_hidden_states).sample
        quantized.save_pretrained(tmp_path / "quantized", max_shard_size="1KB")
        reloaded = Transformer2DModel.from_pretrained(
            tmp_path / "quantized", torch_dtype=torch.bfloat16, device_map={"": "cuda"}
        )
        assert not any(m.use_convrot for m in reloaded.modules() if is_entropack_linear(m))
        with torch.no_grad():
            reloaded_output = reloaded(x, encoder_hidden_states=encoder_hidden_states).sample
        torch.testing.assert_close(output, reloaded_output, rtol=0, atol=0)

    def test_modules_to_not_convert_is_honoured(self, tmp_path):
        model = tiny_model()
        model.save_pretrained(tmp_path / "base")
        quantized = Transformer2DModel.from_pretrained(
            tmp_path / "base",
            quantization_config=EntroPackConfig(target_bpp=4.0, modules_to_not_convert=["to_q"]),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        attention = quantized.transformer_blocks[0].attn1
        assert type(attention.to_q) is nn.Linear
        assert is_entropack_linear(attention.to_k)

    def test_dequantize_restores_linear_layers(self, tmp_path):
        model = tiny_model()
        model.save_pretrained(tmp_path / "base")
        quantized = Transformer2DModel.from_pretrained(
            tmp_path / "base",
            quantization_config=EntroPackConfig(target_bpp=4.0, linear_type="int8"),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        restored = quantized.dequantize()
        assert not restored.is_quantized
        assert not any(is_entropack_linear(module) for module in restored.modules())

        x, encoder_hidden_states = model_inputs("cuda")
        with torch.no_grad():
            output = restored(x, encoder_hidden_states=encoder_hidden_states).sample
        assert torch.isfinite(output).all()
