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
"""CPU-only tests for quantization config logic that doesn't need a backend installed."""

from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.quantization_config import AutoRoundConfig


class TestAutoRoundConfigMerge:
    def test_only_backend_is_overridden_from_args(self):
        checkpoint_config = AutoRoundConfig(bits=4, group_size=32, sym=False, backend="torch")
        args_config = AutoRoundConfig(backend="auto")

        merged = DiffusersAutoQuantizer.merge_quantization_configs(checkpoint_config, args_config)

        assert merged.backend == "auto"
        assert merged.bits == 4
        assert merged.group_size == 32
        assert merged.sym is False
