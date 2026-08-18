# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import json
import os
from unittest import mock

import pytest
import torch
from huggingface_hub import hf_hub_download

from ...testing_utils import torch_device


def backend_memory_allocated(device: str) -> int:
    """
    Bytes currently allocated on `device`. `tests/testing_utils.py` only exposes the *peak* allocation, which cannot
    show memory being released. Skips on backends that do not implement `memory_allocated()` (e.g. mps).
    """
    device_module = getattr(torch, torch.device(device).type)
    if not hasattr(device_module, "memory_allocated"):
        pytest.skip(f"`memory_allocated()` is not implemented for {device}.")
    return device_module.memory_allocated()


def patch_free_memory(free_bytes: int, total_bytes: int = 80 * 1024):
    """
    Simulate `free_bytes` of free device memory on whichever backend module (cuda/xpu/...) backs `torch_device`.

    `mem_get_info` returns `(free, total)` and is the single point where `AutoOffloadStrategy` learns how much memory
    is available, so patching it makes offloading decisions deterministic instead of dependent on the real free memory
    of the test hardware (an 80GB GPU never runs low on a handful of KB-sized models).
    """
    device_type = torch.device(torch_device).type
    device_module = getattr(torch, device_type, torch.cuda)
    return mock.patch.object(device_module, "mem_get_info", return_value=(free_bytes, total_bytes))


def get_specified_components(path_or_repo_id, cache_dir=None):
    """
    The component names a `modular_model_index.json` actually points at a checkpoint for. Returns `None` when the
    index cannot be fetched, which callers treat as "skip the comparison".
    """
    if os.path.isdir(path_or_repo_id):
        config_path = os.path.join(path_or_repo_id, "modular_model_index.json")
    else:
        try:
            config_path = hf_hub_download(
                repo_id=path_or_repo_id,
                filename="modular_model_index.json",
                local_dir=cache_dir,
            )
        except Exception:
            return None

    with open(config_path) as f:
        config = json.load(f)

    components = set()
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool)):
            continue
        for entry in v:
            if isinstance(entry, dict) and (entry.get("repo") or entry.get("pretrained_model_name_or_path")):
                components.add(k)
                break
    return components
