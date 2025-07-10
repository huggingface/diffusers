#!/usr/bin/env python3

# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

# this script dumps information about the environment

import os
import platform
import sys


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Python version:", sys.version)

print("OS platform:", platform.platform())
print("OS architecture:", platform.machine())
try:
    import psutil

    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024**3)
    available_gb = vm.available / (1024**3)
    print(f"Total RAM:     {total_gb:.2f} GB")
    print(f"Available RAM: {available_gb:.2f} GB")
except ImportError:
    pass

try:
    import torch

    print("Torch version:", torch.__version__)
    print("Cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Cuda version:", torch.version.cuda)
        print("CuDNN version:", torch.backends.cudnn.version())
        print("Number of GPUs available:", torch.cuda.device_count())
        device_properties = torch.cuda.get_device_properties(0)
        total_memory = device_properties.total_memory / (1024**3)
        print(f"CUDA memory: {total_memory} GB")

    print("XPU available:", hasattr(torch, "xpu") and torch.xpu.is_available())
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("XPU model:", torch.xpu.get_device_properties(0).name)
        print("XPU compiler version:", torch.version.xpu)
        print("Number of XPUs available:", torch.xpu.device_count())
        device_properties = torch.xpu.get_device_properties(0)
        total_memory = device_properties.total_memory / (1024**3)
        print(f"XPU memory: {total_memory} GB")


except ImportError:
    print("Torch version:", None)

try:
    import transformers

    print("transformers version:", transformers.__version__)
except ImportError:
    print("transformers version:", None)
