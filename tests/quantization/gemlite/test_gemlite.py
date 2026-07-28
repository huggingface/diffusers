import contextlib
import gc
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.gemlite.gemlite_quantizer import (
    _GemLiteDiffusersProcessor,
    _is_in_skip_modules,
    _normalize_torch_device,
)
from diffusers.quantizers.quantization_config import GemLiteConfig, QuantizationMethod
from diffusers.utils import get_module_from_name, is_gemlite_available

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    require_accelerate,
    require_gemlite,
    require_gemlite_version_greater_or_equal,
    require_torch_gpu,
    torch_device,
)


_GEMLITE_SERIALIZED_STATE_NAMES = {"W_q", "scales", "zeros", "metadata", "orig_shape", "meta_scale"}


enable_full_determinism()


if is_gemlite_available() and torch.cuda.is_available():
    from gemlite.core import DType, GemLiteLinearTriton


def _get_gemlite_config(**kwargs):
    config = {
        "bits": 2,
        "group_size": 16,
        "packing_bitwidth": 8,
        "input_dtype": "fp16",
        "output_dtype": "fp16",
        "scales_dtype": "fp32",
        "zeros_dtype": "fp32",
    }
    config.update(kwargs)
    return GemLiteConfig(**config)


def _create_packed_gemlite_state_dict(
    in_features=64,
    out_features=32,
    w_nbits=4,
    group_size=64,
    packing_bitwidth=32,
    scales_dtype=torch.float16,
    zeros_dtype=torch.float16,
    device=torch_device,
    prefix="",
):
    source_layer = GemLiteLinearTriton(
        W_nbits=w_nbits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
    )
    weight = (
        torch.arange(out_features * in_features, dtype=torch.int32, device=device)
        .remainder(2**w_nbits)
        .to(torch.uint8)
    )
    scales = torch.ones((out_features, in_features // group_size), dtype=scales_dtype, device=device)
    zeros = torch.full(
        (out_features, in_features // group_size),
        (2**w_nbits - 1) // 2,
        dtype=zeros_dtype,
        device=device,
    )

    source_layer.pack(weight, scales, zeros, bias=None, packing_bitwidth=packing_bitwidth)

    return {f"{prefix}.{name}" if prefix else name: value for name, value in source_layer.state_dict().items()}


def _save_packed_gemlite_transformer(transformer, save_directory):
    group_size = 32
    quantized_fqns = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear) and module.in_features == group_size
    ]
    quantization_config = GemLiteConfig(
        bits=8,
        group_size=group_size,
        packing_bitwidth=8,
        input_dtype="fp16",
        output_dtype="fp16",
        scales_dtype="fp32",
        zeros_dtype="fp32",
        quantized_fqns=quantized_fqns,
    )

    from gemlite.helper import A16W8_INT8

    for name in quantized_fqns:
        parent, child_name = get_module_from_name(transformer, name)
        linear = getattr(parent, child_name)
        setattr(
            parent,
            child_name,
            A16W8_INT8(device=str(linear.weight.device), dtype=linear.weight.dtype).from_linear(linear),
        )

    transformer.register_to_config(quantization_config=quantization_config)
    transformer.hf_quantizer = DiffusersAutoQuantizer.from_config(quantization_config, pre_quantized=True)
    transformer.save_pretrained(save_directory, safe_serialization=True)

    return quantized_fqns


class GemLiteConfigTest(unittest.TestCase):
    def test_config_defaults(self):
        config = _get_gemlite_config()

        self.assertEqual(config.quant_method, QuantizationMethod.GEMLITE)
        self.assertEqual(config.compute_dtype, torch.float16)
        self.assertEqual(
            config.to_diff_dict(),
            {
                "quant_method": QuantizationMethod.GEMLITE,
                "bits": 2,
                "group_size": 16,
                "packing_bitwidth": 8,
                "input_dtype": "fp16",
                "output_dtype": "fp16",
                "scales_dtype": "fp32",
                "zeros_dtype": "fp32",
            },
        )

    def test_config_requires_serialized_layout(self):
        with self.assertRaisesRegex(ValueError, "require serialized layout fields"):
            GemLiteConfig()

    def test_config_accepts_64_bit_packing(self):
        config = _get_gemlite_config(packing_bitwidth=64)

        self.assertEqual(config.packing_bitwidth, 64)

    def test_config_rejects_invalid_serialized_layout(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _get_gemlite_config(bits=0)
        with self.assertRaisesRegex(ValueError, "Unsupported GemLite `packing_bitwidth`"):
            _get_gemlite_config(packing_bitwidth=4)
        with self.assertRaisesRegex(ValueError, "must be divisible by `bits`"):
            _get_gemlite_config(bits=3)
        with self.assertRaisesRegex(ValueError, "Unsupported GemLite `bits`"):
            _get_gemlite_config(bits=32, packing_bitwidth=32)
        with self.assertRaisesRegex(ValueError, "Unsupported GemLite serialized dtype"):
            _get_gemlite_config(input_dtype="fp8")

    def test_config_from_dict(self):
        config = DiffusersAutoQuantizer.from_dict(
            {
                "quant_method": "gemlite",
                "compute_dtype": "bfloat16",
                "modules_to_not_convert": ["proj_out"],
                "bits": 2,
                "group_size": 16,
                "packing_bitwidth": 8,
                "input_dtype": "fp16",
                "output_dtype": "fp16",
                "scales_dtype": "fp32",
                "zeros_dtype": "fp32",
            }
        )

        self.assertIsInstance(config, GemLiteConfig)
        self.assertEqual(config.compute_dtype, torch.bfloat16)
        self.assertEqual(config.modules_to_not_convert, ["proj_out"])

    def test_config_round_trip(self):
        config = GemLiteConfig(
            compute_dtype=torch.bfloat16,
            modules_to_not_convert=["proj_out"],
            format="gemlite-int2-ternary-g128",
            bits=2,
            group_size=128,
            packing_bitwidth=8,
            solver="ternary",
            input_dtype="fp16",
            output_dtype="fp16",
            scales_dtype="fp32",
            zeros_dtype="fp32",
            quantized_fqns=["blocks.0.proj"],
        )

        restored = DiffusersAutoQuantizer.from_dict(config.to_dict())

        self.assertIsInstance(restored, GemLiteConfig)
        self.assertEqual(restored.compute_dtype, torch.bfloat16)
        self.assertEqual(restored.modules_to_not_convert, ["proj_out"])
        self.assertEqual(restored.format, "gemlite-int2-ternary-g128")
        self.assertEqual(restored.bits, 2)
        self.assertEqual(restored.group_size, 128)
        self.assertEqual(restored.packing_bitwidth, 8)
        self.assertEqual(restored.solver, "ternary")
        self.assertEqual(restored.input_dtype, "fp16")
        self.assertEqual(restored.output_dtype, "fp16")
        self.assertEqual(restored.scales_dtype, "fp32")
        self.assertEqual(restored.zeros_dtype, "fp32")
        self.assertEqual(restored.quantized_fqns, ["blocks.0.proj"])

    def test_quantizer_uses_compute_dtype_when_torch_dtype_is_not_provided(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(compute_dtype=torch.bfloat16))

        self.assertEqual(quantizer.update_torch_dtype(None), torch.bfloat16)
        self.assertEqual(quantizer.update_torch_dtype(torch.bfloat16), torch.bfloat16)

    def test_quantizer_rejects_mismatched_torch_dtype(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(compute_dtype=torch.bfloat16))

        with self.assertRaisesRegex(ValueError, "must match `GemLiteConfig.compute_dtype`"):
            quantizer.update_torch_dtype(torch.float16)

    def test_quantizer_disables_parallel_loading(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config())

        self.assertFalse(quantizer.supports_parallel_loading)

    def test_quantizer_rejects_disk_offloading(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config())

        with self.assertRaisesRegex(ValueError, "does not support disk offloading"):
            quantizer.validate_environment(device_map={"transformer_blocks.0": "disk"})

    def test_quantizer_rejects_unquantized_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "only supports loading pre-quantized checkpoints"):
            DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=False)


class GemLiteQuantizerHelperTest(unittest.TestCase):
    def test_is_in_skip_modules_matches_exact_names_and_children(self):
        modules_to_not_convert = ["proj_out", "blocks.0"]

        self.assertTrue(_is_in_skip_modules("proj_out", modules_to_not_convert))
        self.assertTrue(_is_in_skip_modules("blocks.0.attn.to_q", modules_to_not_convert))
        self.assertFalse(_is_in_skip_modules("proj_output", modules_to_not_convert))
        self.assertFalse(_is_in_skip_modules("blocks.01.attn.to_q", modules_to_not_convert))

    def test_normalize_torch_device_handles_int_device_map_values(self):
        self.assertEqual(_normalize_torch_device(0), torch.device("cuda:0"))
        self.assertEqual(_normalize_torch_device("cpu"), torch.device("cpu"))
        self.assertEqual(_normalize_torch_device(torch.device("meta")), torch.device("meta"))


@require_gemlite
@require_torch_gpu
class GemLiteQuantizerEnvironmentTest(unittest.TestCase):
    @contextlib.contextmanager
    def _mock_module_availability(self, gemlite, cuda, old_gemlite):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch("diffusers.quantizers.gemlite.gemlite_quantizer.is_gemlite_available", return_value=gemlite)
            )
            stack.enter_context(
                mock.patch(
                    "diffusers.quantizers.gemlite.gemlite_quantizer.is_gemlite_version",
                    return_value=old_gemlite,
                )
            )
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=cuda))
            yield

    def test_validate_environment_checks_pre_quantized_serialized_state(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)

        with self._mock_module_availability(gemlite=True, cuda=True, old_gemlite=False):
            quantizer.validate_environment()

    def test_validate_environment_rejects_missing_gemlite(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)

        with self._mock_module_availability(gemlite=False, cuda=True, old_gemlite=False):
            with self.assertRaisesRegex(ImportError, "requires the gemlite library"):
                quantizer.validate_environment()

    def test_validate_environment_rejects_old_gemlite_version(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)

        with self._mock_module_availability(gemlite=True, cuda=True, old_gemlite=True):
            with self.assertRaisesRegex(ImportError, "requires gemlite>=0.6.0"):
                quantizer.validate_environment()


@require_gemlite
@require_gemlite_version_greater_or_equal("0.6.0")
@require_torch_gpu
class GemLiteQuantizerTest(unittest.TestCase):
    def test_processor_replaces_gemlite_linears(self):
        from gemlite.helper import patch_model

        from diffusers import FluxTransformer2DModel

        skip_patterns = [
            "proj_out",
            "x_embedder",
            "context_embedder",
            "time_text_embed",
            "time_guidance_embed",
            "norm_out",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
            "single_stream_modulation",
        ]
        model = FluxTransformer2DModel(
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        ).to(torch_device)
        original_linear_names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        skipped_linear_names = {name for name in original_linear_names if _is_in_skip_modules(name, skip_patterns)}

        processor = _GemLiteDiffusersProcessor(
            GemLiteConfig(
                bits=2,
                group_size=16,
                packing_bitwidth=8,
                input_dtype="fp16",
                output_dtype="fp16",
                scales_dtype="fp32",
                zeros_dtype="fp32",
            ),
            skip_patterns,
        )
        patch_model(
            model,
            device=torch_device,
            processor=processor,
            skip_modules=[],
        )

        self.assertEqual(sum(isinstance(module, GemLiteLinearTriton) for module in model.modules()), 20)
        for name, module in model.named_modules():
            if name in skipped_linear_names:
                self.assertIs(type(module), nn.Linear)
            elif name in original_linear_names:
                self.assertIsInstance(module, GemLiteLinearTriton)

    def test_processor_preserves_bias_state_key(self):
        from gemlite.helper import patch_model

        model = nn.Sequential(nn.Linear(32, 32, bias=True), nn.Linear(32, 32, bias=False)).to(torch_device)

        processor = _GemLiteDiffusersProcessor(
            GemLiteConfig(
                bits=2,
                group_size=16,
                packing_bitwidth=8,
                input_dtype="fp16",
                output_dtype="fp16",
                scales_dtype="fp32",
                zeros_dtype="fp32",
            ),
            [],
        )
        patch_model(
            model,
            device=torch_device,
            processor=processor,
            skip_modules=[],
        )

        self.assertEqual(sum(isinstance(module, GemLiteLinearTriton) for module in model.modules()), 2)
        state_dict = model.state_dict()
        assert _GEMLITE_SERIALIZED_STATE_NAMES.issubset(
            {name.removeprefix("0.") for name in state_dict if name.startswith("0.")}
        )
        self.assertIn("0.bias", state_dict)
        self.assertNotIn("1.bias", state_dict)

    @require_accelerate
    def test_prequantized_auto_device_map_uses_packed_state_size(self):
        from accelerate.utils import compute_module_sizes

        from diffusers.models.model_loading_utils import _determine_device_map

        class GemLiteDeviceMapTestModel(ModelMixin, ConfigMixin):
            _no_split_modules = []

            @register_to_config
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(128, 32, bias=False)

        packed_state = _create_packed_gemlite_state_dict(
            in_features=128,
            out_features=32,
            w_nbits=2,
            group_size=128,
            packing_bitwidth=8,
            scales_dtype=torch.float32,
            zeros_dtype=torch.float32,
            device=torch_device,
        )
        packed_size = sum(value.numel() * value.element_size() for value in packed_state.values())

        quantizer = DiffusersAutoQuantizer.from_config(
            GemLiteConfig(
                format="gemlite-int2-ternary-g128",
                bits=2,
                group_size=128,
                packing_bitwidth=8,
                solver="ternary",
                input_dtype="fp16",
                output_dtype="fp16",
                scales_dtype="fp32",
                zeros_dtype="fp32",
                quantized_fqns=["proj"],
            ),
            pre_quantized=True,
        )
        gemlite_model = GemLiteDeviceMapTestModel().to("meta")
        quantizer.preprocess_model(gemlite_model, device_map="auto", keep_in_fp32_modules=[])
        placeholder_state_dict = gemlite_model.state_dict()
        for name, value in packed_state.items():
            placeholder = placeholder_state_dict[f"proj.{name}"]
            self.assertEqual(placeholder.shape, value.shape)
            self.assertEqual(placeholder.dtype, value.dtype)

        module_sizes = compute_module_sizes(
            gemlite_model,
            dtype=torch.float16,
            special_dtypes=quantizer.get_special_dtypes_update(gemlite_model, torch.float16),
        )
        self.assertEqual(module_sizes["proj"], packed_size)
        with self.assertWarnsRegex(UserWarning, "Current model requires .* bytes of buffer for offloaded layers"):
            gemlite_device_map = _determine_device_map(
                gemlite_model,
                "auto",
                {0: packed_size // 2, "cpu": packed_size * 2},
                torch.float16,
                hf_quantizer=quantizer,
            )

        self.assertNotIn(0, gemlite_device_map.values())

    def test_process_model_before_weight_loading_replaces_pre_quantized_linears(self):
        from diffusers import FluxTransformer2DModel

        skip_patterns = [
            "proj_out",
            "x_embedder",
            "context_embedder",
            "time_text_embed",
            "time_guidance_embed",
            "norm_out",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
            "single_stream_modulation",
        ]
        quantization_config = GemLiteConfig(
            modules_to_not_convert=skip_patterns[:4],
            bits=2,
            group_size=16,
            packing_bitwidth=8,
            input_dtype="fp16",
            output_dtype="fp16",
            scales_dtype="fp32",
            zeros_dtype="fp32",
        )
        quantizer = DiffusersAutoQuantizer.from_config(quantization_config, pre_quantized=True)
        model = FluxTransformer2DModel(
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        ).to(torch_device)
        original_linears = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        skipped_linear_names = {name for name in original_linears if _is_in_skip_modules(name, skip_patterns)}

        quantizer._process_model_before_weight_loading(
            model, device_map=None, keep_in_fp32_modules=[*skip_patterns[4:], None]
        )

        processed_modules = dict(model.named_modules())
        for name, linear in original_linears.items():
            if name in skipped_linear_names:
                self.assertIs(processed_modules[name], linear)
            else:
                self.assertIsInstance(processed_modules[name], GemLiteLinearTriton)
        self.assertEqual(quantizer.modules_to_not_convert, skip_patterns)
        self.assertIs(model.config.quantization_config, quantization_config)

    def test_check_if_quantized_param_pre_quantized(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)

        plain_model = nn.Sequential(nn.Linear(32, 32, bias=False)).to(torch_device)
        self.assertFalse(
            quantizer.check_if_quantized_param(
                plain_model, torch.ones(4, 4, dtype=torch.uint8, device=torch_device), "0.W_q", {}
            )
        )

        gemlite_model = nn.Sequential(GemLiteLinearTriton()).to(torch_device)
        self.assertFalse(
            quantizer.check_if_quantized_param(gemlite_model, torch.ones(32, 32, device=torch_device), "0.weight", {})
        )
        self.assertTrue(
            quantizer.check_if_quantized_param(
                gemlite_model, torch.ones(4, 4, dtype=torch.uint8, device=torch_device), "0.W_q", {}
            )
        )

    def test_create_quantized_param_loads_pre_quantized_state(self):
        in_features = 64
        out_features = 32
        gemlite_state_dict = _create_packed_gemlite_state_dict(in_features, out_features, device=torch_device)
        model = nn.Sequential(GemLiteLinearTriton()).to(torch_device)
        model[0]._gemlite_loaded_param_names = set()
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)

        for name, value in gemlite_state_dict.items():
            with self.subTest(name=name):
                quantizer.create_quantized_param(model, value, f"0.{name}", torch_device)

                loaded_value = getattr(model[0], name)
                self.assertIsInstance(loaded_value, nn.Parameter)
                self.assertIs(model[0]._parameters[name], loaded_value)
                self.assertEqual(loaded_value.device, value.device)
                self.assertEqual(loaded_value.dtype, value.dtype)
                self.assertFalse(loaded_value.requires_grad)
                self.assertTrue(torch.equal(loaded_value, value))

        self.assertEqual(model[0]._gemlite_loaded_param_names, set(gemlite_state_dict))

    def test_create_quantized_param_preserves_serialized_dtypes(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)
        model = nn.Sequential(GemLiteLinearTriton()).to(torch_device)
        model[0]._gemlite_loaded_param_names = set()
        serialized_state = {
            "0.W_q": torch.arange(4, dtype=torch.float16, device=torch_device).reshape(2, 2).to(torch.float8_e5m2),
            "0.scales": torch.tensor([[0.12345679], [0.9876543]], dtype=torch.float32, device=torch_device),
        }

        for param_name, original_value in serialized_state.items():
            with self.subTest(param_name=param_name):
                quantizer.create_quantized_param(
                    model,
                    original_value.to(torch.float16),
                    param_name,
                    torch_device,
                    state_dict=serialized_state,
                )

                loaded_value = getattr(model[0], param_name.removeprefix("0."))
                self.assertEqual(loaded_value.dtype, original_value.dtype)
                self.assertTrue(torch.equal(loaded_value, original_value))

    def test_get_state_dict_and_metadata_sets_data_contiguous_for_serialization(self):
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)
        w_q = torch.arange(12, dtype=torch.uint8, device=torch_device).reshape(3, 4).t()
        metadata = torch.tensor([0, 8, 4, 255, 1, 1, 1, 0, 0, 0, 2, 0], dtype=torch.int32, device=torch_device)
        state_dict = {
            "0.W_q": w_q,
            "0.metadata": metadata,
            "0.orig_shape": torch.tensor([3, 4], dtype=torch.int32, device=torch_device),
        }

        serialized_state_dict, _ = quantizer.get_state_dict_and_metadata(state_dict, safe_serialization=True)

        self.assertFalse(w_q.is_contiguous())
        self.assertEqual(metadata[-1].item(), 0)
        self.assertTrue(serialized_state_dict["0.W_q"].is_contiguous())
        self.assertTrue(torch.equal(serialized_state_dict["0.W_q"], w_q))
        self.assertEqual(serialized_state_dict["0.metadata"][-1].item(), 1)

    def test_create_quantized_param_rejects_non_gemlite_serialized_name(self):
        model = nn.Sequential(GemLiteLinearTriton()).to(torch_device)
        quantizer = DiffusersAutoQuantizer.from_config(_get_gemlite_config(), pre_quantized=True)

        with self.assertRaisesRegex(ValueError, "is not a GemLite serialized tensor"):
            quantizer.create_quantized_param(model, torch.ones(32, 64, device=torch_device), "0.weight", torch_device)


@nightly
@require_gemlite
@require_gemlite_version_greater_or_equal("0.6.0")
@require_torch_gpu
@require_accelerate
class GemLiteKrea2TransformerIntegrationTests(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-krea2-modular-pipe"
    torch_dtype = torch.float16
    maximum_quantized_memory_fraction = 0.8

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_matches_unquantized_transformer(self):
        from diffusers import Krea2Transformer2DModel

        reference_transformer = Krea2Transformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            dtype=self.torch_dtype,
        ).to(torch_device)
        unquantized_transformer_memory = reference_transformer.get_memory_footprint()
        inputs = {
            "hidden_states": torch.randn((1, 4, 16), device=torch_device, dtype=self.torch_dtype),
            "encoder_hidden_states": torch.randn((1, 4, 12, 16), device=torch_device, dtype=self.torch_dtype),
            "timestep": torch.tensor([0.5], device=torch_device, dtype=self.torch_dtype),
            "position_ids": torch.zeros((8, 3), device=torch_device),
            "encoder_attention_mask": torch.tensor([[True, True, True, False]], device=torch_device),
        }
        with torch.inference_mode():
            reference_output = reference_transformer(**inputs).sample

        with tempfile.TemporaryDirectory() as model_dir:
            quantized_fqns = _save_packed_gemlite_transformer(reference_transformer, model_dir)
            self.assertTrue(quantized_fqns)

            del reference_transformer
            gc.collect()
            backend_empty_cache(torch_device)

            transformer = Krea2Transformer2DModel.from_pretrained(
                model_dir,
                dtype=self.torch_dtype,
                device_map={"": torch_device},
            )

        self.assertTrue(any(isinstance(module, GemLiteLinearTriton) for module in transformer.modules()))
        self.assertLessEqual(
            transformer.get_memory_footprint(),
            unquantized_transformer_memory * self.maximum_quantized_memory_fraction,
        )
        with torch.inference_mode():
            gemlite_output = transformer(**inputs).sample

        torch.testing.assert_close(gemlite_output, reference_output, rtol=5e-2, atol=5e-2)


@nightly
@require_gemlite
@require_gemlite_version_greater_or_equal("0.6.0")
@require_torch_gpu
@require_accelerate
class GemLiteFluxPipelineIntegrationTests(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-flux-pipe"
    torch_dtype = torch.float16
    maximum_quantized_memory_fraction = 0.8

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_matches_unquantized_pipeline(self):
        from diffusers import FluxPipeline, FluxTransformer2DModel

        pipe_inputs = {
            "prompt": "a photo of a cat",
            "height": 32,
            "width": 32,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        reference_pipe = FluxPipeline.from_pretrained(self.model_id, dtype=self.torch_dtype).to(torch_device)
        unquantized_transformer_memory = reference_pipe.transformer.get_memory_footprint()
        reference_pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            reference_output = reference_pipe(
                **pipe_inputs,
                generator=torch.Generator(device=torch_device).manual_seed(0),
            ).images

        with tempfile.TemporaryDirectory() as model_dir:
            quantized_fqns = _save_packed_gemlite_transformer(reference_pipe.transformer, model_dir)
            self.assertTrue(quantized_fqns)

            del reference_pipe
            gc.collect()
            backend_empty_cache(torch_device)

            transformer = FluxTransformer2DModel.from_pretrained(
                model_dir,
                dtype=self.torch_dtype,
                device_map={"": torch_device},
            )

        self.assertTrue(any(isinstance(module, GemLiteLinearTriton) for module in transformer.modules()))
        self.assertLessEqual(
            transformer.get_memory_footprint(),
            unquantized_transformer_memory * self.maximum_quantized_memory_fraction,
        )
        gemlite_pipe = FluxPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            dtype=self.torch_dtype,
        ).to(torch_device)
        gemlite_pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            gemlite_output = gemlite_pipe(
                **pipe_inputs,
                generator=torch.Generator(device=torch_device).manual_seed(0),
            ).images

        np.testing.assert_allclose(gemlite_output, reference_output, rtol=5e-2, atol=5e-2)
