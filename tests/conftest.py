# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
import warnings
from os.path import abspath, dirname, join


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "big_accelerator: marks tests as requiring big accelerator resources")
    config.addinivalue_line("markers", "lora: marks tests for LoRA/PEFT functionality")
    config.addinivalue_line("markers", "ip_adapter: marks tests for IP Adapter functionality")
    config.addinivalue_line("markers", "training: marks tests for training functionality")
    config.addinivalue_line("markers", "attention: marks tests for attention processor functionality")
    config.addinivalue_line("markers", "memory: marks tests for memory optimization functionality")
    config.addinivalue_line("markers", "cpu_offload: marks tests for CPU offloading functionality")
    config.addinivalue_line("markers", "group_offload: marks tests for group offloading functionality")
    config.addinivalue_line("markers", "compile: marks tests for torch.compile functionality")
    config.addinivalue_line("markers", "single_file: marks tests for single file checkpoint loading")
    config.addinivalue_line("markers", "quantization: marks tests for quantization functionality")
    config.addinivalue_line("markers", "bitsandbytes: marks tests for BitsAndBytes quantization functionality")
    config.addinivalue_line("markers", "quanto: marks tests for Quanto quantization functionality")
    config.addinivalue_line("markers", "torchao: marks tests for TorchAO quantization functionality")
    config.addinivalue_line("markers", "gguf: marks tests for GGUF quantization functionality")
    config.addinivalue_line("markers", "modelopt: marks tests for NVIDIA ModelOpt quantization functionality")
    config.addinivalue_line("markers", "context_parallel: marks tests for context parallel inference functionality")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "nightly: mark test as nightly")


def pytest_addoption(parser):
    from .testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from .testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
