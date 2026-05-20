# Copyright 2025 - 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Smoke tests for the Quark diffusers quantizer.

The full Quark feature matrix (FP8/INT4/SVDQuant on SDXL/SD3/Flux) is
covered in Quark's own CI on AMD hardware.  These tests cover the bits
that live in diffusers itself: registration, config parsing, and an
on-the-fly weight-only path on a tiny UNet.
"""

from __future__ import annotations

import gc
import unittest

import pytest

from diffusers.utils import is_quark_available, is_torch_available
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    require_accelerator,
    torch_device,
)


if is_torch_available():
    import torch

    from diffusers import UNet2DModel
    from diffusers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
    )
    from diffusers.quantizers.quantization_config import (
        QuantizationMethod,
        QuarkConfig,
    )
    from diffusers.quantizers.quark import QuarkDiffusersQuantizer


requires_quark = pytest.mark.skipif(not is_quark_available(), reason="amd-quark not installed")
enable_full_determinism()


# ---------------------------------------------------------------------------
# Registration / enum / config plumbing.  Pass without amd-quark installed.
# ---------------------------------------------------------------------------


class QuarkRegistrationTest(unittest.TestCase):
    def test_quant_method_enum_has_quark(self):
        self.assertEqual(QuantizationMethod.QUARK, "quark")

    def test_auto_quantizer_mapping_contains_quark(self):
        self.assertIn("quark", AUTO_QUANTIZER_MAPPING)
        self.assertIs(AUTO_QUANTIZER_MAPPING["quark"], QuarkDiffusersQuantizer)

    def test_auto_quantization_config_mapping_contains_quark(self):
        self.assertIn("quark", AUTO_QUANTIZATION_CONFIG_MAPPING)
        self.assertIs(AUTO_QUANTIZATION_CONFIG_MAPPING["quark"], QuarkConfig)


# ---------------------------------------------------------------------------
# Quark-side parsing.  Requires amd-quark.
# ---------------------------------------------------------------------------


@requires_quark
class QuarkConfigParseTest(unittest.TestCase):
    def _native_config_dict(self) -> dict:
        from quark.torch.export.config.config import JsonExporterConfig
        from quark.torch.quantization.config.config import (
            Int8PerTensorSpec,
            QConfig,
            QLayerConfig,
        )

        weight_spec = Int8PerTensorSpec(
            observer_method="min_max",
            symmetric=True,
            scale_type="float",
            round_method="half_even",
            is_dynamic=False,
        ).to_quantization_spec()
        qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))
        body = qconfig.to_dict()
        body["quant_method"] = "quark"
        body["export"] = JsonExporterConfig().__dict__
        return body

    def test_native_config_round_trips(self):
        body = self._native_config_dict()
        cfg = QuarkConfig(**body)
        self.assertEqual(cfg.quant_method, QuantizationMethod.QUARK)
        self.assertEqual(cfg.custom_mode, "quark")
        self.assertFalse(cfg.legacy)
        from quark.torch.quantization.config.config import QConfig

        self.assertIsInstance(cfg.quant_config, QConfig)


# ---------------------------------------------------------------------------
# On-the-fly weight-only round trip on a tiny UNet.
# ---------------------------------------------------------------------------


@nightly
@requires_quark
@require_accelerator
class QuarkOnTheFlyWeightOnlyTest(unittest.TestCase):
    def _make_tiny_unet(self) -> "UNet2DModel":
        return UNet2DModel(
            sample_size=32,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )

    def setUp(self):
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        backend_empty_cache(torch_device)
        gc.collect()

    def test_int8_weight_only_load_time(self):
        from quark.torch.export.config.config import JsonExporterConfig
        from quark.torch.quantization.config.config import (
            Int8PerTensorSpec,
            QConfig,
            QLayerConfig,
        )

        weight_spec = Int8PerTensorSpec(
            observer_method="min_max",
            symmetric=True,
            scale_type="float",
            round_method="half_even",
            is_dynamic=False,
        ).to_quantization_spec()
        qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

        body = qconfig.to_dict()
        body["quant_method"] = "quark"
        body["export"] = JsonExporterConfig().__dict__

        # We don't go through from_pretrained here to keep the test
        # self-contained -- exercise the quantizer hooks against a fresh
        # in-memory model instead.
        from diffusers.quantizers.quark import QuarkDiffusersQuantizer

        torch.manual_seed(0)
        model = self._make_tiny_unet().to(torch_device).eval()
        quark_config = QuarkConfig(**body)
        quantizer = QuarkDiffusersQuantizer(quark_config)
        quantizer.pre_quantized = False  # type: ignore[attr-defined]

        quantizer._process_model_before_weight_loading(model)
        # check_if_quantized_param returns False for the on-the-fly path,
        # so weight loading is a no-op for our purposes; jump straight to
        # the post-load step.
        quantizer._process_model_after_weight_loading(model)

        sample = torch.randn(1, 1, 32, 32, device=torch_device)
        timestep = torch.tensor([1.0], device=torch_device)
        with torch.no_grad():
            out = model(sample, timestep).sample
        self.assertEqual(out.shape, (1, 1, 32, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_activation_quantized_config_rejected(self):
        from quark.torch.quantization.config.config import (
            Int8PerTensorSpec,
            QConfig,
            QLayerConfig,
        )

        weight_spec = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()
        act_spec = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()
        qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec, input_tensors=act_spec))

        body = qconfig.to_dict()
        body["quant_method"] = "quark"

        from diffusers.quantizers.quark import QuarkDiffusersQuantizer

        model = self._make_tiny_unet().to(torch_device).eval()
        quark_config = QuarkConfig(**body)
        quantizer = QuarkDiffusersQuantizer(quark_config)
        quantizer.pre_quantized = False  # type: ignore[attr-defined]

        with self.assertRaisesRegex(NotImplementedError, "weight-only"):
            quantizer._process_model_before_weight_loading(model)
