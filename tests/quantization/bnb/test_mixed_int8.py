# coding=utf-8
# Copyright 2026 The HuggingFace Team Inc.
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
import gc

import pytest
import torch

from diffusers import BitsAndBytesConfig
from diffusers.utils import logging

from ...testing_utils import (
    CaptureLogger,
    backend_empty_cache,
    is_bitsandbytes,
    is_bitsandbytes_available,
    is_quantization,
    require_accelerate,
    require_bitsandbytes_version_greater,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_bitsandbytes_available():
    from diffusers.quantizers.bitsandbytes.utils import replace_with_bnb_linear


# Model-level BitsAndBytes tests live in `tests/models/testing_utils/quantization.py` and
# pipeline-level ones in `tests/pipelines/testing_utils/quantization.py`. This module covers
# backend behavior that fits neither: utility warnings.
@is_quantization
@is_bitsandbytes
@require_bitsandbytes_version_greater("0.43.2")
@require_accelerate
@require_torch
@require_torch_accelerator
@slow
class TestBnB8bitBasic:
    @pytest.fixture(autouse=True)
    def _setup_basic(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_bnb_8bit_logs_warning_for_no_quantization(self):
        model_with_no_linear = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3), torch.nn.ReLU())
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger = logging.get_logger("diffusers.quantizers.bitsandbytes.utils")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            _ = replace_with_bnb_linear(model_with_no_linear, quantization_config=quantization_config)
        assert (
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            in cap_logger.out
        )
