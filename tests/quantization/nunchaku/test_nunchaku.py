import gc
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from diffusers import ConfigMixin, ModelMixin, NunchakuLiteQuantizationConfig
from diffusers.configuration_utils import register_to_config
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.quantizers.nunchaku.nunchaku_quantizer import NunchakuLiteQuantizer

from ...testing_utils import (
    backend_empty_cache,
    nightly,
    require_accelerator,
    require_kernels_version_greater_or_equal,
    torch_device,
)


class TinyPretrainedModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 128, bias=True)
        self.linear2 = torch.nn.Linear(64, 128, bias=False)


def _state_dict(precision="int4"):
    state_dict = {
        "linear1.bias": torch.randn(128, dtype=torch.bfloat16),
        "linear1.proj_down": torch.randn(64, 4, dtype=torch.bfloat16),
        "linear1.proj_up": torch.randn(128, 4, dtype=torch.bfloat16),
        "linear1.qweight": torch.randint(-8, 8, (128, 32), dtype=torch.int8),
        "linear1.smooth_factor": torch.randn(64, dtype=torch.bfloat16),
        "linear2.qweight": torch.randint(-8, 8, (32, 32), dtype=torch.int32),
        "linear2.wscales": torch.randn(1, 128, dtype=torch.bfloat16),
        "linear2.wzeros": torch.randn(1, 128, dtype=torch.bfloat16),
    }
    if precision == "nvfp4":
        state_dict["linear1.wcscales"] = torch.randn(128, dtype=torch.bfloat16)
        state_dict["linear1.wscales"] = torch.empty(4, 128, dtype=torch.float8_e4m3fn)
        state_dict["linear1.wtscale"] = torch.randn(1, dtype=torch.bfloat16)
    else:
        state_dict["linear1.wscales"] = torch.randn(1, 128, dtype=torch.bfloat16)
    return state_dict


def _compact_config():
    return {
        "svdq_w4a4": {
            "precision": "nvfp4",
            "group_size": 16,
            "rank": 4,
            "targets": ["linear1"],
        },
        "awq_w4a16": {
            "precision": "int4",
            "group_size": 64,
            "targets": ["linear2"],
        },
    }


@nightly
@require_accelerator
@require_kernels_version_greater_or_equal("0.9.0")
class NunchakuLiteCudaKernelsTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_awq_cuda_kernels(self):
        if torch_device != "cuda":
            self.skipTest("Nunchaku Lite CUDA kernels test requires CUDA device")

        torch.manual_seed(0)
        from diffusers.quantizers.nunchaku.utils import AWQW4A16Linear

        layer = AWQW4A16Linear(64, 128, bias=True, group_size=64, torch_dtype=torch.bfloat16, device=torch_device)
        layer.qweight.data = torch.randint(-8, 8, layer.qweight.shape, dtype=torch.int32, device=torch_device)
        layer.wscales.data = torch.rand(layer.wscales.shape, dtype=torch.bfloat16, device=torch_device)
        layer.wzeros.data = torch.rand(layer.wzeros.shape, dtype=torch.bfloat16, device=torch_device)
        layer.bias.data.zero_()

        for shape in [(1, 8, 64), (1, 16, 64)]:
            x = torch.randn(shape, dtype=torch.bfloat16, device=torch_device)
            with torch.no_grad():
                output = layer(x)

            self.assertEqual(output.shape, (*shape[:-1], 128))
            self.assertFalse(torch.isnan(output).any())


class NunchakuLiteBasicTests(unittest.TestCase):
    model_cls = TinyPretrainedModel

    def test_compact_config_round_trips_dtype_and_targets(self):
        quantization_config = NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **_compact_config())
        config_dict = quantization_config.to_dict()

        self.assertEqual(config_dict["compute_dtype"], "bfloat16")
        self.assertEqual(config_dict["svdq_w4a4"]["precision"], "nvfp4")

        reloaded_config = NunchakuLiteQuantizationConfig.from_dict(config_dict)
        self.assertEqual(reloaded_config.compute_dtype, torch.bfloat16)
        self.assertEqual(reloaded_config.svdq_w4a4["targets"], ["linear1"])

    def test_nvfp4_environment_requires_blackwell_cuda(self):
        quantization_config = NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **_compact_config())
        quantizer = NunchakuLiteQuantizer(quantization_config)

        with (
            patch("diffusers.quantizers.nunchaku.nunchaku_quantizer.is_kernels_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with self.assertRaisesRegex(ValueError, "Blackwell or newer NVIDIA GPU"):
                quantizer.validate_environment()

    def test_nvfp4_environment_allows_blackwell_cuda(self):
        quantization_config = NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **_compact_config())
        quantizer = NunchakuLiteQuantizer(quantization_config)

        with (
            patch("diffusers.quantizers.nunchaku.nunchaku_quantizer.is_kernels_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            quantizer.validate_environment()

    def test_int4_environment_requires_turing_cuda(self):
        compact_config = _compact_config()
        compact_config["svdq_w4a4"]["precision"] = "int4"
        compact_config["svdq_w4a4"]["group_size"] = 64
        quantization_config = NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **compact_config)
        quantizer = NunchakuLiteQuantizer(quantization_config)

        with (
            patch("diffusers.quantizers.nunchaku.nunchaku_quantizer.is_kernels_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
        ):
            with self.assertRaisesRegex(ValueError, "Turing or newer NVIDIA GPU"):
                quantizer.validate_environment()

    def test_int4_environment_allows_turing_cuda(self):
        compact_config = _compact_config()
        compact_config["svdq_w4a4"]["precision"] = "int4"
        compact_config["svdq_w4a4"]["group_size"] = 64
        quantization_config = NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **compact_config)
        quantizer = NunchakuLiteQuantizer(quantization_config)

        with (
            patch("diffusers.quantizers.nunchaku.nunchaku_quantizer.is_kernels_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            quantizer.validate_environment()

    def test_environment_requires_cuda(self):
        compact_config = _compact_config()
        compact_config["svdq_w4a4"]["precision"] = "int4"
        compact_config["svdq_w4a4"]["group_size"] = 64
        quantization_config = NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **compact_config)
        quantizer = NunchakuLiteQuantizer(quantization_config)

        with (
            patch("diffusers.quantizers.nunchaku.nunchaku_quantizer.is_kernels_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "CUDA-capable NVIDIA GPU"):
                quantizer.validate_environment()

    @require_kernels_version_greater_or_equal("0.9.0")
    def test_compact_config_replaces_svdq_and_awq_without_state_dict(self):
        from diffusers.quantizers.nunchaku.utils import AWQW4A16Linear, SVDQW4A4Linear

        model = self.model_cls()
        quantizer = DiffusersAutoQuantizer.from_config(
            NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16, **_compact_config())
        )

        quantizer.preprocess_model(model)

        self.assertIsInstance(model.linear1, SVDQW4A4Linear)
        self.assertIsInstance(model.linear2, AWQW4A16Linear)
        self.assertEqual(model.linear1.precision, "nvfp4")
        self.assertEqual(model.linear1.rank, 4)
        self.assertIsNotNone(model.linear1.bias)
        self.assertIsNone(model.linear2.bias)

    @require_kernels_version_greater_or_equal("0.9.0")
    def test_nunchaku_lite_loads_with_from_pretrained(self):
        from diffusers.quantizers.nunchaku.utils import AWQW4A16Linear, SVDQW4A4Linear

        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.model_cls()
            model.save_config(tmpdir)

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path) as handle:
                config = json.load(handle)

            compact_config = _compact_config()
            config["quantization_config"] = NunchakuLiteQuantizationConfig(
                compute_dtype=torch.bfloat16, **compact_config
            ).to_dict()

            with open(config_path, "w") as handle:
                json.dump(config, handle)

            svdq_config = compact_config["svdq_w4a4"]
            precision = "nvfp4" if svdq_config["precision"] == "nvfp4" else "int4"
            save_file(_state_dict(precision=precision), os.path.join(tmpdir, "diffusion_pytorch_model.safetensors"))

            loaded_model = self.model_cls.from_pretrained(tmpdir)

        self.assertIsInstance(loaded_model.linear1, SVDQW4A4Linear)
        self.assertIsInstance(loaded_model.linear2, AWQW4A16Linear)
        self.assertEqual(loaded_model.linear1.precision, "nvfp4")


if __name__ == "__main__":
    unittest.main()
