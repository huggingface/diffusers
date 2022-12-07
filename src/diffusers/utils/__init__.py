# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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


import os

from packaging import version

from .. import __version__
from .deprecation_utils import deprecate
from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    is_accelerate_available,
    is_flax_available,
    is_inflect_available,
    is_librosa_available,
    is_modelcards_available,
    is_onnx_available,
    is_safetensors_available,
    is_scipy_available,
    is_tf_available,
    is_torch_available,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    requires_backends,
)
from .logging import get_logger
from .outputs import BaseOutput
from .pil_utils import PIL_INTERPOLATION


if is_torch_available():
    from .testing_utils import (
        floats_tensor,
        load_hf_numpy,
        load_image,
        load_numpy,
        parse_flag_from_env,
        require_torch_gpu,
        slow,
        torch_all_close,
        torch_device,
    )


logger = get_logger(__name__)


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "diffusers")


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
DIFFUSERS_CACHE = default_cache_path
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))

_COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS = [
    "DDIMScheduler",
    "DDPMScheduler",
    "PNDMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "HeunDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler",
]


def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = (
                "This example requires a source install from HuggingFace diffusers (see "
                "`https://huggingface.co/docs/diffusers/installation#install-from-source`),"
            )
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(error_message)
