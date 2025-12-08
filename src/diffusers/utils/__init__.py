# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from .constants import (
    CONFIG_NAME,
    DEFAULT_HF_PARALLEL_LOADING_WORKERS,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_DYNAMIC_MODULE_NAME,
    FLAX_WEIGHTS_NAME,
    GGUF_FILE_EXTENSION,
    HF_ENABLE_PARALLEL_LOADING,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    MIN_PEFT_VERSION,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    SAFETENSORS_WEIGHTS_NAME,
    USE_PEFT_BACKEND,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from .deprecation_utils import _maybe_remap_transformers_class, deprecate
from .doc_utils import replace_example_docstring
from .dynamic_modules_utils import get_class_from_dynamic_module
from .export_utils import export_to_gif, export_to_obj, export_to_ply, export_to_video
from .hub_utils import (
    PushToHubMixin,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    extract_commit_hash,
    http_user_agent,
)
from .import_utils import (
    BACKENDS_MAPPING,
    DIFFUSERS_SLOW_IMPORT,
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_accelerate_available,
    is_accelerate_version,
    is_aiter_available,
    is_aiter_version,
    is_better_profanity_available,
    is_bitsandbytes_available,
    is_bitsandbytes_version,
    is_bs4_available,
    is_cosmos_guardrail_available,
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_flash_attn_version,
    is_flax_available,
    is_ftfy_available,
    is_gguf_available,
    is_gguf_version,
    is_google_colab,
    is_hf_hub_version,
    is_hpu_available,
    is_inflect_available,
    is_invisible_watermark_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_kernels_available,
    is_kornia_available,
    is_librosa_available,
    is_matplotlib_available,
    is_nltk_available,
    is_note_seq_available,
    is_nvidia_modelopt_available,
    is_nvidia_modelopt_version,
    is_onnx_available,
    is_opencv_available,
    is_optimum_quanto_available,
    is_optimum_quanto_version,
    is_peft_available,
    is_peft_version,
    is_pytorch_retinaface_available,
    is_safetensors_available,
    is_sageattention_available,
    is_sageattention_version,
    is_scipy_available,
    is_sentencepiece_available,
    is_tensorboard_available,
    is_timm_available,
    is_torch_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_version,
    is_torch_xla_available,
    is_torch_xla_version,
    is_torchao_available,
    is_torchao_version,
    is_torchsde_available,
    is_torchvision_available,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    is_wandb_available,
    is_xformers_available,
    is_xformers_version,
    requires_backends,
)
from .loading_utils import get_module_from_name, get_submodule_by_name, load_image, load_video
from .logging import get_logger
from .outputs import BaseOutput
from .peft_utils import (
    check_peft_version,
    delete_adapter_layers,
    get_adapter_name,
    get_peft_kwargs,
    recurse_remove_peft_layers,
    scale_lora_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
    unscale_lora_layers,
)
from .pil_utils import PIL_INTERPOLATION, make_image_grid, numpy_to_pil, pt_to_pil
from .remote_utils import remote_decode
from .state_dict_utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
    state_dict_all_zero,
)
from .typing_utils import _get_detailed_type, _is_valid_type


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
