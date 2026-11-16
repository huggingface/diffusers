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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import ast
import sys
import warnings
from os.path import abspath, dirname, join

import pytest


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def marker_decorators():
    """Map each `is_*` decorator in `tests/testing_utils.py` to the pytest marker it applies.

    Only that file is read: it is where the marker decorators are defined. The `testing_utils/` packages
    under `tests/models/` and `tests/pipelines/` hold the tester mixins that *use* these decorators, and the
    per-pipeline `testing_utils.py` files hold test helpers; neither defines markers. Read from the source
    with `ast` rather than imported, so `utils/tests_fetcher.py` can call this without importing torch.
    Matches `def is_x(test_case): return pytest.mark.<marker>(test_case)`, so with

        def is_lora(test_case):
            return pytest.mark.lora(test_case)

        def is_torch_compile(test_case):
            return pytest.mark.compile(test_case)

    in `tests/testing_utils.py` the result is `{"is_lora": "lora", "is_torch_compile": "compile"}`. A new `is_*`
    decorator is picked up automatically; nothing else needs registering.
    """
    tree = ast.parse(open(join(dirname(__file__), "testing_utils.py"), encoding="utf-8").read())
    decorators = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("is_"):
            continue
        for ret in ast.walk(node):
            if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call):
                func = ret.value.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "mark"
                ):
                    decorators[node.name] = func.attr
    return decorators


# Every marker an `is_*` decorator applies takes a test out of the `core` set. CI runs each feature in its
# own job (`-m lora`, ...) and everything else as `-m core`, so a test carrying none of these is marked
# `core` at collection time. `utils/tests_fetcher.py` uses the same set to decide which jobs to schedule.
NON_CORE_MARKERS = set(marker_decorators().values())


def pytest_configure(config):
    config.addinivalue_line("markers", "core: marks tests carrying none of the feature markers (added automatically)")
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
    config.addinivalue_line("markers", "torchao: marks tests for TorchAO quantization functionality")
    config.addinivalue_line("markers", "gguf: marks tests for GGUF quantization functionality")
    config.addinivalue_line("markers", "modelopt: marks tests for NVIDIA ModelOpt quantization functionality")
    config.addinivalue_line("markers", "sdnq: marks tests for SDNQ quantization functionality")
    config.addinivalue_line("markers", "bfl: marks tests for BFL (fp8r / nvfp4) quantization functionality")
    config.addinivalue_line("markers", "context_parallel: marks tests for context parallel inference functionality")
    config.addinivalue_line("markers", "tensor_parallel: marks tests for tensor parallel inference functionality")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "nightly: mark test as nightly")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Mark every test carrying none of `NON_CORE_MARKERS` as `core`.

    CI runs the feature jobs with `-m lora`, `-m memory`, ... and everything else with `-m core`. A positive
    marker is the only way to say "everything else" to pytest: `-m "not (lora or memory or ...)"` would have
    to list every feature marker and silently go stale. Adding the marker here, to every collected test
    without a feature marker, keeps `core` exhaustive by construction. `tryfirst` runs this before pytest's
    own `-m` deselection, which happens in the same hook.
    """
    for item in items:
        if not any(m.name in NON_CORE_MARKERS for m in item.iter_markers()):
            item.add_marker("core")


def pytest_addoption(parser):
    from .testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from .testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
