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
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.util
import inspect
import operator as op
import os
import sys
from collections import OrderedDict, defaultdict
from functools import lru_cache as cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging.version import Version, parse

from . import logging


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
try:
    _package_map = importlib_metadata.packages_distributions()  # load-once to avoid expensive calls
except Exception:
    _package_map = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()
DIFFUSERS_SLOW_IMPORT = os.environ.get("DIFFUSERS_SLOW_IMPORT", "FALSE").upper()
DIFFUSERS_SLOW_IMPORT = DIFFUSERS_SLOW_IMPORT in ENV_VARS_TRUE_VALUES

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_is_google_colab = "google.colab" in sys.modules or any(k.startswith("COLAB_") for k in os.environ)


def _is_package_available(pkg_name: str, get_dist_name: bool = False) -> Tuple[bool, str]:
    global _package_map
    pkg_exists = importlib.util.find_spec(pkg_name) is not None
    pkg_version = "N/A"

    if pkg_exists:
        if _package_map is None:
            _package_map = defaultdict(list)
            try:
                # Fallback for Python < 3.10
                for dist in importlib_metadata.distributions():
                    _top_level_declared = (dist.read_text("top_level.txt") or "").split()
                    # Infer top-level package names from file structure
                    _inferred_opt_names = {
                        f.parts[0] if len(f.parts) > 1 else inspect.getmodulename(f) for f in (dist.files or [])
                    } - {None}
                    _top_level_inferred = filter(lambda name: "." not in name, _inferred_opt_names)
                    for pkg in _top_level_declared or _top_level_inferred:
                        _package_map[pkg].append(dist.metadata["Name"])
            except Exception as _:
                pass
        try:
            if get_dist_name and pkg_name in _package_map and _package_map[pkg_name]:
                if len(_package_map[pkg_name]) > 1:
                    logger.warning(
                        f"Multiple distributions found for package {pkg_name}. Picked distribution: {_package_map[pkg_name][0]}"
                    )
                pkg_name = _package_map[pkg_name][0]
            pkg_version = importlib_metadata.version(pkg_name)
            logger.debug(f"Successfully imported {pkg_name} version {pkg_version}")
        except (ImportError, importlib_metadata.PackageNotFoundError):
            pkg_exists = False

    return pkg_exists, pkg_version


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch")

else:
    logger.info("Disabling PyTorch because USE_TORCH is set")
    _torch_available = False
    _torch_version = "N/A"

_jax_version = "N/A"
_flax_version = "N/A"
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
    if _flax_available:
        try:
            _jax_version = importlib_metadata.version("jax")
            _flax_version = importlib_metadata.version("flax")
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _flax_available = False
else:
    _flax_available = False

if USE_SAFETENSORS in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _safetensors_available, _safetensors_version = _is_package_available("safetensors")

else:
    logger.info("Disabling Safetensors because USE_SAFETENSORS is set")
    _safetensors_available = False

_onnxruntime_version = "N/A"
_onnx_available = importlib.util.find_spec("onnxruntime") is not None
if _onnx_available:
    candidates = (
        "onnxruntime",
        "onnxruntime-cann",
        "onnxruntime-directml",
        "ort_nightly_directml",
        "onnxruntime-gpu",
        "ort_nightly_gpu",
        "onnxruntime-migraphx",
        "onnxruntime-openvino",
        "onnxruntime-qnn",
        "onnxruntime-rocm",
        "onnxruntime-training",
        "onnxruntime-vitisai",
    )
    _onnxruntime_version = None
    # For the metadata, we have to look for both onnxruntime and onnxruntime-x
    for pkg in candidates:
        try:
            _onnxruntime_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _onnx_available = _onnxruntime_version is not None
    if _onnx_available:
        logger.debug(f"Successfully imported onnxruntime version {_onnxruntime_version}")

# (sayakpaul): importlib.util.find_spec("opencv-python") returns None even when it's installed.
# _opencv_available = importlib.util.find_spec("opencv-python") is not None
try:
    candidates = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    _opencv_version = None
    for pkg in candidates:
        try:
            _opencv_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _opencv_available = _opencv_version is not None
    if _opencv_available:
        logger.debug(f"Successfully imported cv2 version {_opencv_version}")
except importlib_metadata.PackageNotFoundError:
    _opencv_available = False

_bs4_available = importlib.util.find_spec("bs4") is not None
try:
    # importlib metadata under different name
    _bs4_version = importlib_metadata.version("beautifulsoup4")
    logger.debug(f"Successfully imported ftfy version {_bs4_version}")
except importlib_metadata.PackageNotFoundError:
    _bs4_available = False

_invisible_watermark_available = importlib.util.find_spec("imwatermark") is not None
try:
    _invisible_watermark_version = importlib_metadata.version("invisible-watermark")
    logger.debug(f"Successfully imported invisible-watermark version {_invisible_watermark_version}")
except importlib_metadata.PackageNotFoundError:
    _invisible_watermark_available = False

_torch_xla_available, _torch_xla_version = _is_package_available("torch_xla")
_torch_npu_available, _torch_npu_version = _is_package_available("torch_npu")
_transformers_available, _transformers_version = _is_package_available("transformers")
_hf_hub_available, _hf_hub_version = _is_package_available("huggingface_hub")
_kernels_available, _kernels_version = _is_package_available("kernels")
_inflect_available, _inflect_version = _is_package_available("inflect")
_unidecode_available, _unidecode_version = _is_package_available("unidecode")
_k_diffusion_available, _k_diffusion_version = _is_package_available("k_diffusion")
_note_seq_available, _note_seq_version = _is_package_available("note_seq")
_wandb_available, _wandb_version = _is_package_available("wandb")
_tensorboard_available, _tensorboard_version = _is_package_available("tensorboard")
_compel_available, _compel_version = _is_package_available("compel")
_sentencepiece_available, _sentencepiece_version = _is_package_available("sentencepiece")
_torchsde_available, _torchsde_version = _is_package_available("torchsde")
_peft_available, _peft_version = _is_package_available("peft")
_torchvision_available, _torchvision_version = _is_package_available("torchvision")
_matplotlib_available, _matplotlib_version = _is_package_available("matplotlib")
_timm_available, _timm_version = _is_package_available("timm")
_bitsandbytes_available, _bitsandbytes_version = _is_package_available("bitsandbytes")
_imageio_available, _imageio_version = _is_package_available("imageio")
_ftfy_available, _ftfy_version = _is_package_available("ftfy")
_scipy_available, _scipy_version = _is_package_available("scipy")
_librosa_available, _librosa_version = _is_package_available("librosa")
_accelerate_available, _accelerate_version = _is_package_available("accelerate")
_xformers_available, _xformers_version = _is_package_available("xformers")
_gguf_available, _gguf_version = _is_package_available("gguf")
_torchao_available, _torchao_version = _is_package_available("torchao")
_bitsandbytes_available, _bitsandbytes_version = _is_package_available("bitsandbytes")
_optimum_quanto_available, _optimum_quanto_version = _is_package_available("optimum", get_dist_name=True)
_pytorch_retinaface_available, _pytorch_retinaface_version = _is_package_available("pytorch_retinaface")
_better_profanity_available, _better_profanity_version = _is_package_available("better_profanity")
_nltk_available, _nltk_version = _is_package_available("nltk")
_cosmos_guardrail_available, _cosmos_guardrail_version = _is_package_available("cosmos_guardrail")
_sageattention_available, _sageattention_version = _is_package_available("sageattention")
_flash_attn_available, _flash_attn_version = _is_package_available("flash_attn")
_flash_attn_3_available, _flash_attn_3_version = _is_package_available("flash_attn_3")
_kornia_available, _kornia_version = _is_package_available("kornia")
_nvidia_modelopt_available, _nvidia_modelopt_version = _is_package_available("modelopt", get_dist_name=True)


def is_torch_available():
    return _torch_available


def is_torch_xla_available():
    return _torch_xla_available


def is_torch_npu_available():
    return _torch_npu_available


def is_flax_available():
    return _flax_available


def is_transformers_available():
    return _transformers_available


def is_inflect_available():
    return _inflect_available


def is_unidecode_available():
    return _unidecode_available


def is_onnx_available():
    return _onnx_available


def is_opencv_available():
    return _opencv_available


def is_scipy_available():
    return _scipy_available


def is_librosa_available():
    return _librosa_available


def is_xformers_available():
    return _xformers_available


def is_accelerate_available():
    return _accelerate_available


def is_kernels_available():
    return _kernels_available


def is_k_diffusion_available():
    return _k_diffusion_available


def is_note_seq_available():
    return _note_seq_available


def is_wandb_available():
    return _wandb_available


def is_tensorboard_available():
    return _tensorboard_available


def is_compel_available():
    return _compel_available


def is_ftfy_available():
    return _ftfy_available


def is_bs4_available():
    return _bs4_available


def is_torchsde_available():
    return _torchsde_available


def is_invisible_watermark_available():
    return _invisible_watermark_available


def is_peft_available():
    return _peft_available


def is_torchvision_available():
    return _torchvision_available


def is_matplotlib_available():
    return _matplotlib_available


def is_safetensors_available():
    return _safetensors_available


def is_bitsandbytes_available():
    return _bitsandbytes_available


def is_google_colab():
    return _is_google_colab


def is_sentencepiece_available():
    return _sentencepiece_available


def is_imageio_available():
    return _imageio_available


def is_gguf_available():
    return _gguf_available


def is_torchao_available():
    return _torchao_available


def is_optimum_quanto_available():
    return _optimum_quanto_available


def is_nvidia_modelopt_available():
    return _nvidia_modelopt_available


def is_timm_available():
    return _timm_available


def is_pytorch_retinaface_available():
    return _pytorch_retinaface_available


def is_better_profanity_available():
    return _better_profanity_available


def is_nltk_available():
    return _nltk_available


def is_cosmos_guardrail_available():
    return _cosmos_guardrail_available


def is_hpu_available():
    return all(importlib.util.find_spec(lib) for lib in ("habana_frameworks", "habana_frameworks.torch"))


def is_sageattention_available():
    return _sageattention_available


def is_flash_attn_available():
    return _flash_attn_available


def is_flash_attn_3_available():
    return _flash_attn_3_available


def is_kornia_available():
    return _kornia_available


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
"""

# docstyle-ignore
INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

# docstyle-ignore
ONNX_IMPORT_ERROR = """
{0} requires the onnxruntime library but it was not found in your environment. You can install it with pip: `pip
install onnxruntime`
"""

# docstyle-ignore
OPENCV_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with pip: `pip
install opencv-python`
"""

# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip: `pip install
scipy`
"""

# docstyle-ignore
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library but it was not found in your environment.  Checkout the instructions on the
installation page: https://librosa.org/doc/latest/install.html and follow the ones that match your environment.
"""

# docstyle-ignore
TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""

# docstyle-ignore
UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""

# docstyle-ignore
K_DIFFUSION_IMPORT_ERROR = """
{0} requires the k-diffusion library but it was not found in your environment. You can install it with pip: `pip
install k-diffusion`
"""

# docstyle-ignore
NOTE_SEQ_IMPORT_ERROR = """
{0} requires the note-seq library but it was not found in your environment. You can install it with pip: `pip
install note-seq`
"""

# docstyle-ignore
WANDB_IMPORT_ERROR = """
{0} requires the wandb library but it was not found in your environment. You can install it with pip: `pip
install wandb`
"""

# docstyle-ignore
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
"""


# docstyle-ignore
COMPEL_IMPORT_ERROR = """
{0} requires the compel library but it was not found in your environment. You can install it with pip: `pip install compel`
"""

# docstyle-ignore
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TORCHSDE_IMPORT_ERROR = """
{0} requires the torchsde library but it was not found in your environment. You can install it with pip: `pip install torchsde`
"""

# docstyle-ignore
INVISIBLE_WATERMARK_IMPORT_ERROR = """
{0} requires the invisible-watermark library but it was not found in your environment. You can install it with pip: `pip install invisible-watermark>=0.2.0`
"""

# docstyle-ignore
PEFT_IMPORT_ERROR = """
{0} requires the peft library but it was not found in your environment. You can install it with pip: `pip install peft`
"""

# docstyle-ignore
SAFETENSORS_IMPORT_ERROR = """
{0} requires the safetensors library but it was not found in your environment. You can install it with pip: `pip install safetensors`
"""

# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the sentencepiece library but it was not found in your environment. You can install it with pip: `pip install sentencepiece`
"""


# docstyle-ignore
BITSANDBYTES_IMPORT_ERROR = """
{0} requires the bitsandbytes library but it was not found in your environment. You can install it with pip: `pip install bitsandbytes`
"""

# docstyle-ignore
IMAGEIO_IMPORT_ERROR = """
{0} requires the imageio library and ffmpeg but it was not found in your environment. You can install it with pip: `pip install imageio imageio-ffmpeg`
"""

# docstyle-ignore
GGUF_IMPORT_ERROR = """
{0} requires the gguf library but it was not found in your environment. You can install it with pip: `pip install gguf`
"""

TORCHAO_IMPORT_ERROR = """
{0} requires the torchao library but it was not found in your environment. You can install it with pip: `pip install
torchao`
"""

QUANTO_IMPORT_ERROR = """
{0} requires the optimum-quanto library but it was not found in your environment. You can install it with pip: `pip
install optimum-quanto`
"""

# docstyle-ignore
PYTORCH_RETINAFACE_IMPORT_ERROR = """
{0} requires the pytorch_retinaface library but it was not found in your environment. You can install it with pip: `pip install pytorch_retinaface`
"""

# docstyle-ignore
BETTER_PROFANITY_IMPORT_ERROR = """
{0} requires the better_profanity library but it was not found in your environment. You can install it with pip: `pip install better_profanity`
"""

# docstyle-ignore
NLTK_IMPORT_ERROR = """
{0} requires the nltk library but it was not found in your environment. You can install it with pip: `pip install nltk`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
        ("onnx", (is_onnx_available, ONNX_IMPORT_ERROR)),
        ("opencv", (is_opencv_available, OPENCV_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("transformers", (is_transformers_available, TRANSFORMERS_IMPORT_ERROR)),
        ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("k_diffusion", (is_k_diffusion_available, K_DIFFUSION_IMPORT_ERROR)),
        ("note_seq", (is_note_seq_available, NOTE_SEQ_IMPORT_ERROR)),
        ("wandb", (is_wandb_available, WANDB_IMPORT_ERROR)),
        ("tensorboard", (is_tensorboard_available, TENSORBOARD_IMPORT_ERROR)),
        ("compel", (is_compel_available, COMPEL_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("torchsde", (is_torchsde_available, TORCHSDE_IMPORT_ERROR)),
        ("invisible_watermark", (is_invisible_watermark_available, INVISIBLE_WATERMARK_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("safetensors", (is_safetensors_available, SAFETENSORS_IMPORT_ERROR)),
        ("bitsandbytes", (is_bitsandbytes_available, BITSANDBYTES_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("imageio", (is_imageio_available, IMAGEIO_IMPORT_ERROR)),
        ("gguf", (is_gguf_available, GGUF_IMPORT_ERROR)),
        ("torchao", (is_torchao_available, TORCHAO_IMPORT_ERROR)),
        ("quanto", (is_optimum_quanto_available, QUANTO_IMPORT_ERROR)),
        ("pytorch_retinaface", (is_pytorch_retinaface_available, PYTORCH_RETINAFACE_IMPORT_ERROR)),
        ("better_profanity", (is_better_profanity_available, BETTER_PROFANITY_IMPORT_ERROR)),
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))

    if name in [
        "VersatileDiffusionTextToImagePipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionDualGuidedPipeline",
        "StableDiffusionImageVariationPipeline",
        "UnCLIPPipeline",
    ] and is_transformers_version("<", "4.25.0"):
        raise ImportError(
            f"You need to install `transformers>=4.25` in order to use {name}: \n```\n pip install"
            " --upgrade transformers \n```"
        )

    if name in ["StableDiffusionDepth2ImgPipeline", "StableDiffusionPix2PixZeroPipeline"] and is_transformers_version(
        "<", "4.26.0"
    ):
        raise ImportError(
            f"You need to install `transformers>=4.26` in order to use {name}: \n```\n pip install"
            " --upgrade transformers \n```"
        )


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_") and key not in ["_load_connected_pipes", "_is_onnx"]:
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L338
@cache
def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


@cache
def is_torch_xla_version(operation: str, version: str):
    """
    Compares the current torch_xla version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of torch_xla
    """
    if not is_torch_xla_available:
        return False
    return compare_versions(parse(_torch_xla_version), operation, version)


@cache
def is_transformers_version(operation: str, version: str):
    """
    Compares the current Transformers version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _transformers_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)


@cache
def is_hf_hub_version(operation: str, version: str):
    """
    Compares the current Hugging Face Hub version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _hf_hub_available:
        return False
    return compare_versions(parse(_hf_hub_version), operation, version)


@cache
def is_accelerate_version(operation: str, version: str):
    """
    Compares the current Accelerate version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _accelerate_available:
        return False
    return compare_versions(parse(_accelerate_version), operation, version)


@cache
def is_peft_version(operation: str, version: str):
    """
    Compares the current PEFT version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _peft_available:
        return False
    return compare_versions(parse(_peft_version), operation, version)


@cache
def is_bitsandbytes_version(operation: str, version: str):
    """
    Args:
    Compares the current bitsandbytes version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _bitsandbytes_available:
        return False
    return compare_versions(parse(_bitsandbytes_version), operation, version)


@cache
def is_gguf_version(operation: str, version: str):
    """
    Compares the current Accelerate version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _gguf_available:
        return False
    return compare_versions(parse(_gguf_version), operation, version)


@cache
def is_torchao_version(operation: str, version: str):
    """
    Compares the current torchao version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _torchao_available:
        return False
    return compare_versions(parse(_torchao_version), operation, version)


@cache
def is_k_diffusion_version(operation: str, version: str):
    """
    Compares the current k-diffusion version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _k_diffusion_available:
        return False
    return compare_versions(parse(_k_diffusion_version), operation, version)


@cache
def is_optimum_quanto_version(operation: str, version: str):
    """
    Compares the current Accelerate version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _optimum_quanto_available:
        return False
    return compare_versions(parse(_optimum_quanto_version), operation, version)


@cache
def is_nvidia_modelopt_version(operation: str, version: str):
    """
    Compares the current Nvidia ModelOpt version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _nvidia_modelopt_available:
        return False
    return compare_versions(parse(_nvidia_modelopt_version), operation, version)


@cache
def is_xformers_version(operation: str, version: str):
    """
    Compares the current xformers version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _xformers_available:
        return False
    return compare_versions(parse(_xformers_version), operation, version)


@cache
def is_sageattention_version(operation: str, version: str):
    """
    Compares the current sageattention version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _sageattention_available:
        return False
    return compare_versions(parse(_sageattention_version), operation, version)


@cache
def is_flash_attn_version(operation: str, version: str):
    """
    Compares the current flash-attention version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _flash_attn_available:
        return False
    return compare_versions(parse(_flash_attn_version), operation, version)


def get_objects_from_module(module):
    """
    Returns a dict of object names and values in a module, while skipping private/internal objects

    Args:
        module (ModuleType):
            Module to extract the objects from.

    Returns:
        dict: Dictionary of object names and corresponding values
    """

    objects = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        objects[name] = getattr(module, name)

    return objects


class OptionalDependencyNotAvailable(BaseException):
    """
    An error indicating that an optional dependency of Diffusers was not found in the environment.
    """


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
