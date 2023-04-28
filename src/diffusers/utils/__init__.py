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


import os

from packaging import version

from .. import __version__
from .accelerate_utils import apply_forward_hook
from .constants import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_CACHE,
    DIFFUSERS_DYNAMIC_MODULE_NAME,
    FLAX_WEIGHTS_NAME,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    TEXT_ENCODER_TARGET_MODULES,
    WEIGHTS_NAME,
)
from .deprecation_utils import deprecate
from .doc_utils import replace_example_docstring
from .dynamic_modules_utils import get_class_from_dynamic_module
from .hub_utils import (
    HF_HUB_OFFLINE,
    _add_variant,
    _get_model_file,
    extract_commit_hash,
    http_user_agent,
)
from .import_utils import (
    BACKENDS_MAPPING,
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    OptionalDependencyNotAvailable,
    is_accelerate_available,
    is_accelerate_version,
    is_bs4_available,
    is_flax_available,
    is_ftfy_available,
    is_inflect_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_librosa_available,
    is_note_seq_available,
    is_omegaconf_available,
    is_onnx_available,
    is_safetensors_available,
    is_scipy_available,
    is_tensorboard_available,
    is_tf_available,
    is_torch_available,
    is_torch_version,
    is_torchsde_available,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    is_wandb_available,
    is_xformers_available,
    requires_backends,
)
from .logging import get_logger
from .outputs import BaseOutput
from .pil_utils import PIL_INTERPOLATION, numpy_to_pil, pt_to_pil
from .torch_utils import is_compiled_module, randn_tensor


if is_torch_available():
    from .testing_utils import (
        floats_tensor,
        load_hf_numpy,
        load_image,
        load_numpy,
        load_pt,
        nightly,
        parse_flag_from_env,
        print_tensor_test,
        require_torch_2,
        require_torch_gpu,
        skip_mps,
        slow,
        torch_all_close,
        torch_device,
    )

from .testing_utils import export_to_video


logger = get_logger(__name__)


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
