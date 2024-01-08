# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import os

from huggingface_hub.constants import HF_HOME
from packaging import version

from ..dependency_versions_check import dep_version_check
from .import_utils import ENV_VARS_TRUE_VALUES, is_peft_available, is_transformers_available


MIN_PEFT_VERSION = "0.6.0"
MIN_TRANSFORMERS_VERSION = "4.34.0"
_CHECK_PEFT = os.environ.get("_CHECK_PEFT", "1") in ENV_VARS_TRUE_VALUES


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(HF_HOME, "modules"))
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]

# Below should be `True` if the current version of `peft` and `transformers` are compatible with
# PEFT backend. Will automatically fall back to PEFT backend if the correct versions of the libraries are
# available.
# For PEFT it is has to be greater than or equal to 0.6.0 and for transformers it has to be greater than or equal to 4.34.0.
_required_peft_version = is_peft_available() and version.parse(
    version.parse(importlib.metadata.version("peft")).base_version
) >= version.parse(MIN_PEFT_VERSION)
_required_transformers_version = is_transformers_available() and version.parse(
    version.parse(importlib.metadata.version("transformers")).base_version
) >= version.parse(MIN_TRANSFORMERS_VERSION)

USE_PEFT_BACKEND = _required_peft_version and _required_transformers_version

if USE_PEFT_BACKEND and _CHECK_PEFT:
    dep_version_check("peft")
