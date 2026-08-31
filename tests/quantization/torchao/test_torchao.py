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


import pytest

from diffusers import TorchAoConfig

from ...testing_utils import (
    enable_full_determinism,
    is_quantization,
    is_torchao,
    is_torchao_available,
    require_torch,
    require_torch_accelerator,
    require_torchao_version_greater_or_equal,
)


if is_torchao_available():
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig


enable_full_determinism()


# Model-level TorchAO tests live in `tests/models/testing_utils/quantization.py` and
# pipeline-level ones in `tests/pipelines/testing_utils/quantization.py`. This module covers
# backend behavior that fits neither: config validation and custom device maps with cpu/disk offload.
@is_quantization
@is_torchao
@require_torch
@require_torch_accelerator
@require_torchao_version_greater_or_equal("0.15.0")
class TestTorchAoConfig:
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = TorchAoConfig(Int4WeightOnlyConfig(version=2))
        torchao_orig_config = quantization_config.to_dict()
        assert "quant_type" in torchao_orig_config
        assert "quant_method" in torchao_orig_config

    def test_post_init_check(self):
        """
        Test that non-AOBaseConfig types are rejected
        """
        _ = TorchAoConfig(Int4WeightOnlyConfig())
        with pytest.raises(TypeError):
            _ = TorchAoConfig("int4_weight_only")

        with pytest.raises(TypeError):
            _ = TorchAoConfig(42)

    def test_repr(self):
        """
        Check that there is no error in the repr
        """
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=["conv"])
        quantization_repr = repr(quantization_config)
        assert "TorchAoConfig" in quantization_repr
        assert "torchao" in quantization_repr
