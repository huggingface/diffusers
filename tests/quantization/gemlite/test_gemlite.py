import contextlib
import gc
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.gemlite import GemLiteQuantizer
from diffusers.quantizers.gemlite.gemlite_quantizer import (
    _quantize_linears_on_the_fly,
    _replace_with_gemlite_linear,
)
from diffusers.quantizers.quantization_config import GemLiteConfig, QuantizationMethod
from diffusers.utils import is_gemlite_available, is_torch_available

from ...testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    require_accelerate,
    require_accelerator,
    require_gemlite,
    require_gemlite_version_greater_or_equal,
    torch_device,
)


# TODO(gemlite): bump to the next released version once it ships, so the version gate
# actually protects against installs older than the API these tests rely on.
_GEMLITE_MIN_VERSION = "0.5.1"


enable_full_determinism()


if is_torch_available():
    from ..utils import get_memory_consumption_stat


if is_gemlite_available():
    from gemlite.core import DType, GemLiteLinearTriton
    from gemlite.helper import A16W8_FP8


def _create_packed_gemlite_state_dict(in_features=64, out_features=32, w_nbits=4, group_size=64):
    source_layer = GemLiteLinearTriton(
        W_nbits=w_nbits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
    )
    weight = torch.arange(out_features * in_features, dtype=torch.int32).remainder(2**w_nbits).to(torch.uint8)
    scales = torch.ones((weight.numel() // group_size, 1), dtype=torch.float16)
    zeros = torch.full((weight.numel() // group_size, 1), (2**w_nbits - 1) // 2, dtype=torch.float16)

    source_layer.pack(weight, scales, zeros, bias=None)

    return source_layer.state_dict()


class GemLiteConfigTest(unittest.TestCase):
    def test_config_defaults(self):
        config = GemLiteConfig()

        self.assertEqual(config.quant_method, QuantizationMethod.GEMLITE)
        self.assertEqual(config.compute_dtype, torch.float16)
        self.assertEqual(config.weight_quant_format, "int8")
        self.assertEqual(config.to_diff_dict()["quant_method"], QuantizationMethod.GEMLITE)

    def test_config_from_dict(self):
        config = DiffusersAutoQuantizer.from_dict(
            {"quant_method": "gemlite", "compute_dtype": "bfloat16", "modules_to_not_convert": ["proj_out"]}
        )

        self.assertIsInstance(config, GemLiteConfig)
        self.assertEqual(config.compute_dtype, torch.bfloat16)
        self.assertEqual(config.modules_to_not_convert, ["proj_out"])
        self.assertEqual(config.weight_quant_format, "int8")

    def test_config_round_trip(self):
        config = GemLiteConfig(
            compute_dtype=torch.bfloat16, modules_to_not_convert=["proj_out"], weight_quant_format="fp8"
        )

        restored = DiffusersAutoQuantizer.from_dict(config.to_dict())

        self.assertIsInstance(restored, GemLiteConfig)
        self.assertEqual(restored.compute_dtype, torch.bfloat16)
        self.assertEqual(restored.modules_to_not_convert, ["proj_out"])
        self.assertEqual(restored.weight_quant_format, "fp8")

    def test_config_rejects_unsupported_weight_quant_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported `weight_quant_format='float8'`"):
            GemLiteConfig(weight_quant_format="float8")


class GemLiteAutoQuantizerTest(unittest.TestCase):
    def test_auto_quantizer_mapping(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        self.assertIsInstance(quantizer, GemLiteQuantizer)
        self.assertTrue(quantizer.pre_quantized)


@require_gemlite
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

    def test_validate_environment_accepts_available_environment(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        with self._mock_module_availability(gemlite=True, cuda=True, old_gemlite=False):
            quantizer.validate_environment()

    def test_validate_environment_rejects_missing_gemlite(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        with self._mock_module_availability(gemlite=False, cuda=True, old_gemlite=False):
            with self.assertRaisesRegex(ImportError, "requires the gemlite library"):
                quantizer.validate_environment()

    def test_validate_environment_rejects_old_gemlite_version(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        with self._mock_module_availability(gemlite=True, cuda=True, old_gemlite=True):
            with self.assertRaisesRegex(ImportError, "requires gemlite>=0.5.1"):
                quantizer.validate_environment()

    def test_validate_environment_rejects_broken_gemlite_core_import(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)
        original_import = __import__

        def import_with_broken_gemlite_core(name, *args, **kwargs):
            if name == "gemlite.core":
                raise RuntimeError("broken gemlite core")
            return original_import(name, *args, **kwargs)

        with self._mock_module_availability(gemlite=True, cuda=True, old_gemlite=False):
            with mock.patch("builtins.__import__", side_effect=import_with_broken_gemlite_core):
                with self.assertRaisesRegex(ImportError, "core linear module could not be imported"):
                    quantizer.validate_environment()

    def test_validate_environment_rejects_unsupported_on_the_fly_weight_quant_format(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=False)
        quantizer.weight_quant_format = "int4"

        with self._mock_module_availability(gemlite=True, cuda=True, old_gemlite=False):
            with self.assertRaisesRegex(ValueError, "got 'int4'"):
                quantizer.validate_environment()


@require_gemlite
@require_gemlite_version_greater_or_equal(_GEMLITE_MIN_VERSION)
class GemLiteQuantizerBackendTest(unittest.TestCase):
    def test_check_if_quantized_param_rejects_non_gemlite_modules(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        plain_model = nn.Sequential(nn.Linear(32, 32, bias=False))
        self.assertFalse(
            quantizer.check_if_quantized_param(plain_model, torch.ones(4, 4, dtype=torch.uint8), "0.W_q", {})
        )

        gemlite_model = nn.Sequential(GemLiteLinearTriton())
        self.assertFalse(quantizer.check_if_quantized_param(gemlite_model, torch.ones(32, 32), "0.weight", {}))
        self.assertTrue(
            quantizer.check_if_quantized_param(gemlite_model, torch.ones(4, 4, dtype=torch.uint8), "0.W_q", {})
        )

    def test_replace_skips_dotted_nested_modules(self):
        inner = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False))
        model = nn.Sequential(inner, nn.Linear(32, 32, bias=False))

        replaced = _replace_with_gemlite_linear(model, ["0"])

        self.assertEqual(replaced, 1)
        self.assertIs(type(model[0][0]), nn.Linear)
        self.assertIs(type(model[0][1]), nn.Linear)
        self.assertIsInstance(model[1], GemLiteLinearTriton)

    def test_replace_preserves_bias_state_key(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=True), nn.Linear(32, 32, bias=False))

        replaced = _replace_with_gemlite_linear(model, [])

        self.assertEqual(replaced, 2)
        self.assertIn("0.bias", model.state_dict())
        self.assertNotIn("1.bias", model.state_dict())

    def test_process_before_weight_loading_writes_back_config(self):
        config = GemLiteConfig()
        quantizer = DiffusersAutoQuantizer.from_config(config, pre_quantized=True)

        model = nn.Sequential(nn.Linear(32, 32, bias=False))

        class FakeConfig:
            quantization_config = None

        model.config = FakeConfig()

        quantizer._process_model_before_weight_loading(model, device_map=None)

        self.assertIs(model.config.quantization_config, config)

    def test_uses_upstream_gemlite_module_and_finalizes_loaded_state(self):
        in_features = 64
        out_features = 32
        gemlite_state_dict = _create_packed_gemlite_state_dict(in_features, out_features)

        model = nn.Sequential(nn.Linear(in_features, out_features, bias=False))
        replaced = _replace_with_gemlite_linear(model, [])
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        self.assertEqual(replaced, 1)
        self.assertIsInstance(model[0], GemLiteLinearTriton)

        for name, value in gemlite_state_dict.items():
            quantizer.create_quantized_param(model, value, f"0.{name}", "cpu")

        self.assertEqual(model[0]._gemlite_loaded_param_names, set(gemlite_state_dict))

        quantizer._process_model_after_weight_loading(model)
        loaded_state_dict = model[0].state_dict()
        # `metadata` and `orig_shape` are metadata containers that gemlite's `load_state_dict`
        # converts to Python list/tuple, so they no longer appear in `state_dict()`. Only the
        # real tensor data (W_q, scales, zeros, bias) should round-trip.
        for name in ("W_q", "scales", "zeros"):
            self.assertTrue(torch.equal(loaded_state_dict[name], gemlite_state_dict[name]), name)

    def test_create_quantized_param_preserves_serialized_dtype(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(weight_quant_format="fp8"), pre_quantized=True)
        model = nn.Sequential(GemLiteLinearTriton())
        original_w_q = torch.arange(4, dtype=torch.float16).reshape(2, 2).to(torch.float8_e5m2)

        quantizer.create_quantized_param(
            model,
            original_w_q.to(torch.float16),
            "0.W_q",
            "cpu",
            state_dict={"0.W_q": original_w_q},
        )

        self.assertEqual(model[0].W_q.dtype, torch.float8_e5m2)

    def test_check_if_quantized_param_disabled_on_the_fly(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=False)

        gemlite_model = nn.Sequential(GemLiteLinearTriton())
        # On-the-fly path always returns False: it never interprets state-dict entries as quantized params.
        self.assertFalse(
            quantizer.check_if_quantized_param(gemlite_model, torch.ones(4, 4, dtype=torch.uint8), "0.W_q", {})
        )

    @require_accelerator
    def test_quantize_linears_on_the_fly_replaces_linears(self):
        inner = nn.Sequential(nn.Linear(32, 64, bias=False), nn.Linear(32, 32, bias=False))
        model = nn.Sequential(inner, nn.Linear(64, 32, bias=False)).to(torch_device, dtype=torch.float16)

        quantized = _quantize_linears_on_the_fly(
            model, ["0.1"], compute_dtype=torch.float16, weight_quant_format="int8"
        )

        # Two skipped via "0.1" (matches dotted nested), one under inner[0], one top-level[1].
        self.assertEqual(quantized, 2)
        self.assertIsInstance(model[0][0], GemLiteLinearTriton)
        self.assertIs(type(model[0][1]), nn.Linear)
        self.assertIsInstance(model[1], GemLiteLinearTriton)

    @require_accelerator
    def test_quantize_linears_on_the_fly_supports_fp8(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=False)).to(torch_device, dtype=torch.float16)

        quantized = _quantize_linears_on_the_fly(model, [], compute_dtype=torch.float16, weight_quant_format="fp8")

        self.assertEqual(quantized, 1)
        self.assertIsInstance(model[0], GemLiteLinearTriton)
        self.assertEqual(model[0].W_q.dtype, A16W8_FP8().fp8)

    @require_accelerator
    def test_quantize_linears_on_the_fly_rejects_cpu_model(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=False))

        with self.assertRaisesRegex(ValueError, "requires the model to be on a CUDA or ROCm device"):
            _quantize_linears_on_the_fly(model, [], compute_dtype=torch.float16, weight_quant_format="int8")

    @require_accelerator
    def test_quantize_linears_on_the_fly_skips_small_in_features(self):
        # in_features=16 is below GemLite's MIN_SIZE (32); the layer must be left as nn.Linear.
        model = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.Linear(16, 32, bias=False),
            nn.Linear(16, 16, bias=False),
        ).to(torch_device, dtype=torch.float16)

        quantized = _quantize_linears_on_the_fly(model, [], compute_dtype=torch.float16, weight_quant_format="int8")

        self.assertEqual(quantized, 1)
        self.assertIsInstance(model[0], GemLiteLinearTriton)
        self.assertIs(type(model[1]), nn.Linear)
        self.assertIs(type(model[2]), nn.Linear)


@nightly
@require_gemlite
@require_gemlite_version_greater_or_equal(_GEMLITE_MIN_VERSION)
@require_accelerator
@require_accelerate
class GemLiteBaseTesterMixin:
    model_id = "hf-internal-testing/tiny-flux-transformer"
    pipeline_model_id = "hf-internal-testing/tiny-flux-pipe"
    torch_dtype = torch.float16
    expected_memory_reduction = 1.1
    modules_to_not_convert = ["proj_out"]
    weight_quant_format = "int8"

    @classmethod
    def setUpClass(cls):
        from diffusers import FluxTransformer2DModel

        cls._tmpdir = tempfile.TemporaryDirectory()

        quantization_config = GemLiteConfig(
            compute_dtype=cls.torch_dtype,
            modules_to_not_convert=cls.modules_to_not_convert,
            weight_quant_format=cls.weight_quant_format,
        )
        model = FluxTransformer2DModel.from_pretrained(
            cls.model_id,
            torch_dtype=cls.torch_dtype,
            device_map={"": torch_device},
            quantization_config=quantization_config,
        )
        model.save_pretrained(cls._tmpdir.name)

        del model
        gc.collect()
        backend_empty_cache(torch_device)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()
        gc.collect()
        backend_empty_cache(torch_device)

    def setUp(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def _load_quantized_model(self):
        from diffusers import FluxTransformer2DModel

        return FluxTransformer2DModel.from_pretrained(self._tmpdir.name, torch_dtype=self.torch_dtype)

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4096, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 768),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
            "img_ids": torch.randn((4096, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "txt_ids": torch.randn((512, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "guidance": torch.tensor([3.5]).to(torch_device, self.torch_dtype),
        }

    def test_gemlite_layers(self):
        model = self._load_quantized_model()
        model.to(torch_device)
        skip = set(self.modules_to_not_convert)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert name in skip, f"nn.Linear {name} was not replaced with GemLiteLinearTriton"
            if isinstance(module, GemLiteLinearTriton):
                assert name not in skip, f"skipped module {name} was unexpectedly replaced"

    def test_gemlite_memory_usage(self):
        from diffusers import FluxTransformer2DModel

        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=self.torch_dtype) for k, v in inputs.items() if not isinstance(v, bool)
        }

        unquantized_model = FluxTransformer2DModel.from_pretrained(self.model_id, torch_dtype=self.torch_dtype)
        unquantized_model.to(torch_device)
        unquantized_memory = get_memory_consumption_stat(unquantized_model, inputs)
        del unquantized_model
        gc.collect()
        backend_empty_cache(torch_device)

        quantized_model = self._load_quantized_model()
        quantized_model.to(torch_device)
        quantized_memory = get_memory_consumption_stat(quantized_model, inputs)

        assert unquantized_memory / quantized_memory >= self.expected_memory_reduction

    def test_modules_to_not_convert(self):
        model = self._load_quantized_model()
        model.to(torch_device)
        skip = set(self.modules_to_not_convert)

        matched = {name: module for name, module in model.named_modules() if name in skip}
        assert set(matched) == skip, f"skip entries unmatched: {skip - set(matched)}"
        for name, module in matched.items():
            assert isinstance(module, nn.Linear), f"skipped module {name} is not nn.Linear"
            assert not isinstance(module, GemLiteLinearTriton), f"skipped module {name} was replaced"

    def test_dtype_assignment(self):
        model = self._load_quantized_model()
        with self.assertRaises(ValueError):
            model.to(torch.float16)
        with self.assertRaises(ValueError):
            model.float()
        with self.assertRaises(ValueError):
            model.half()
        model.to(torch_device)

    def test_serialization(self):
        model = self._load_quantized_model()
        inputs = self.get_dummy_inputs()
        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            from diffusers import FluxTransformer2DModel

            saved_model = FluxTransformer2DModel.from_pretrained(tmp_dir, torch_dtype=self.torch_dtype)
            saved_model.to(torch_device)
            with torch.no_grad():
                saved_model_output = saved_model(**inputs)

        assert torch.allclose(model_output.sample, saved_model_output.sample, rtol=1e-5, atol=1e-5)

    def test_model_cpu_offload(self):
        from diffusers import FluxPipeline, FluxTransformer2DModel

        quantization_config = GemLiteConfig(
            compute_dtype=self.torch_dtype,
            # The tiny pipeline has in_features=160 here, which GemLite/Triton cannot compile.
            modules_to_not_convert=self.modules_to_not_convert + ["single_transformer_blocks.0.proj_out"],
            weight_quant_format=self.weight_quant_format,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            self.pipeline_model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
            device_map={"": torch_device},
            quantization_config=quantization_config,
        )
        pipe = FluxPipeline.from_pretrained(
            self.pipeline_model_id, transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)

    def test_on_the_fly_quantization_matches_pre_quantized(self):
        from diffusers import FluxTransformer2DModel

        quantization_config = GemLiteConfig(
            compute_dtype=self.torch_dtype,
            modules_to_not_convert=self.modules_to_not_convert,
            weight_quant_format=self.weight_quant_format,
        )
        on_the_fly = FluxTransformer2DModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map={"": torch_device},
            quantization_config=quantization_config,
        )

        skip = set(self.modules_to_not_convert)
        linear_count = 0
        gemlite_count = 0
        for name, module in on_the_fly.named_modules():
            if isinstance(module, nn.Linear):
                assert name in skip, f"nn.Linear {name} was not quantized on-the-fly"
                linear_count += 1
            if isinstance(module, GemLiteLinearTriton):
                assert name not in skip, f"skipped module {name} was unexpectedly quantized"
                gemlite_count += 1
        assert gemlite_count > 0, "No GemLiteLinearTriton modules produced by on-the-fly quantization"
        assert linear_count == len(skip), f"Expected {len(skip)} retained nn.Linear modules, got {linear_count}"

        # Compare outputs against the saved pre-quantized fixture built from the same GemLite format.
        pre_quantized = self._load_quantized_model().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=self.torch_dtype) for k, v in inputs.items() if not isinstance(v, bool)
        }
        with torch.no_grad():
            on_the_fly_out = on_the_fly(**inputs).sample
            pre_quantized_out = pre_quantized(**inputs).sample
        assert torch.allclose(on_the_fly_out, pre_quantized_out, rtol=1e-3, atol=1e-3)

        del on_the_fly, pre_quantized
        gc.collect()
        backend_empty_cache(torch_device)


class FluxTransformerGemLiteINT8Test(GemLiteBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 1.2
    weight_quant_format = "int8"


class FluxTransformerGemLiteFP8Test(GemLiteBaseTesterMixin, unittest.TestCase):
    weight_quant_format = "fp8"
    expected_memory_reduction = 1.1


if __name__ == "__main__":
    unittest.main()
