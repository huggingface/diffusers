# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import operator as op
import os
import sys
from collections import OrderedDict
from typing import Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging import version
from packaging.version import Version, parse

from . import logging


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_torch_version = "N/A"
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TORCH is set")
    _torch_available = False


_tf_version = "N/A"
if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
            "tensorflow-aarch64",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if version.parse(_tf_version) < version.parse("2"):
            logger.info(f"TensorFlow found but with version {_tf_version}. Diffusers requires version 2 minimum.")
            _tf_available = False
        else:
            logger.info(f"TensorFlow version {_tf_version} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False

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
    _safetensors_available = importlib.util.find_spec("safetensors") is not None
    if _safetensors_available:
        try:
            _safetensors_version = importlib_metadata.version("safetensors")
            logger.info(f"Safetensors version {_safetensors_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _safetensors_available = False
else:
    logger.info("Disabling Safetensors because USE_TF is set")
    _safetensors_available = False

_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False


_inflect_available = importlib.util.find_spec("inflect") is not None
try:
    _inflect_version = importlib_metadata.version("inflect")
    logger.debug(f"Successfully imported inflect version {_inflect_version}")
except importlib_metadata.PackageNotFoundError:
    _inflect_available = False


_unidecode_available = importlib.util.find_spec("unidecode") is not None
try:
    _unidecode_version = importlib_metadata.version("unidecode")
    logger.debug(f"Successfully imported unidecode version {_unidecode_version}")
except importlib_metadata.PackageNotFoundError:
    _unidecode_available = False


_onnxruntime_version = "N/A"
_onnx_available = importlib.util.find_spec("onnxruntime") is not None
if _onnx_available:
    candidates = (
        "onnxruntime",
        "onnxruntime-gpu",
        "ort_nightly_gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
        "ort_nightly_directml",
        "onnxruntime-rocm",
        "onnxruntime-training",
    )
    _onnxruntime_version = None
    # For the metadata, we have to look for both onnxruntime and onnxruntime-gpu
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

_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
    logger.debug(f"Successfully imported scipy version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False

_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    _librosa_version = importlib_metadata.version("librosa")
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    _librosa_available = False

_accelerate_available = importlib.util.find_spec("accelerate") is not None
try:
    _accelerate_version = importlib_metadata.version("accelerate")
    logger.debug(f"Successfully imported accelerate version {_accelerate_version}")
except importlib_metadata.PackageNotFoundError:
    _accelerate_available = False

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    _xformers_version = importlib_metadata.version("xformers")
    if _torch_available:
        import torch

        if version.Version(torch.__version__) < version.Version("1.12"):
            raise ValueError("PyTorch should be >= 1.12")
    logger.debug(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False

_k_diffusion_available = importlib.util.find_spec("k_diffusion") is not None
try:
    _k_diffusion_version = importlib_metadata.version("k_diffusion")
    logger.debug(f"Successfully imported k-diffusion version {_k_diffusion_version}")
except importlib_metadata.PackageNotFoundError:
    _k_diffusion_available = False

_note_seq_available = importlib.util.find_spec("note_seq") is not None
try:
    _note_seq_version = importlib_metadata.version("note_seq")
    logger.debug(f"Successfully imported note-seq version {_note_seq_version}")
except importlib_metadata.PackageNotFoundError:
    _note_seq_available = False

_wandb_available = importlib.util.find_spec("wandb") is not None
try:
    _wandb_version = importlib_metadata.version("wandb")
    logger.debug(f"Successfully imported wandb version {_wandb_version }")
except importlib_metadata.PackageNotFoundError:
    _wandb_available = False

_omegaconf_available = importlib.util.find_spec("omegaconf") is not None
try:
    _omegaconf_version = importlib_metadata.version("omegaconf")
    logger.debug(f"Successfully imported omegaconf version {_omegaconf_version}")
except importlib_metadata.PackageNotFoundError:
    _omegaconf_available = False

_tensorboard_available = importlib.util.find_spec("tensorboard")
try:
    _tensorboard_version = importlib_metadata.version("tensorboard")
    logger.debug(f"Successfully imported tensorboard version {_tensorboard_version}")
except importlib_metadata.PackageNotFoundError:
    _tensorboard_available = False


_compel_available = importlib.util.find_spec("compel")
try:
    _compel_version = importlib_metadata.version("compel")
    logger.debug(f"Successfully imported compel version {_compel_version}")
except importlib_metadata.PackageNotFoundError:
    _compel_available = False


_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False


_bs4_available = importlib.util.find_spec("bs4") is not None
try:
    # importlib metadata under different name
    _bs4_version = importlib_metadata.version("beautifulsoup4")
    logger.debug(f"Successfully imported ftfy version {_bs4_version}")
except importlib_metadata.PackageNotFoundError:
    _bs4_available = False

_torchsde_available = importlib.util.find_spec("torchsde") is not None
try:
    _torchsde_version = importlib_metadata.version("torchsde")
    logger.debug(f"Successfully imported torchsde version {_torchsde_version}")
except importlib_metadata.PackageNotFoundError:
    _torchsde_available = False

_invisible_watermark_available = importlib.util.find_spec("imwatermark") is not None
try:
    _invisible_watermark_version = importlib_metadata.version("invisible-watermark")
    logger.debug(f"Successfully imported invisible-watermark version {_invisible_watermark_version}")
except importlib_metadata.PackageNotFoundError:
    _invisible_watermark_available = False


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


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


def is_k_diffusion_available():
    return _k_diffusion_available


def is_note_seq_available():
    return _note_seq_available


def is_wandb_available():
    return _wandb_available


def is_omegaconf_available():
    return _omegaconf_available


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
OMEGACONF_IMPORT_ERROR = """
{0} requires the omegaconf library but it was not found in your environment. You can install it with pip: `pip
install omegaconf`
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
        ("omegaconf", (is_omegaconf_available, OMEGACONF_IMPORT_ERROR)),
        ("tensorboard", (is_tensorboard_available, TENSORBOARD_IMPORT_ERROR)),
        ("compel", (is_compel_available, COMPEL_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("torchsde", (is_torchsde_available, TORCHSDE_IMPORT_ERROR)),
        ("invisible_watermark", (is_invisible_watermark_available, INVISIBLE_WATERMARK_IMPORT_ERROR)),
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
    Args:
    Compares a library version to some requirement using a given operation.
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
def is_torch_version(operation: str, version: str):
    """
    Args:
    Compares the current PyTorch version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


def is_transformers_version(operation: str, version: str):
    """
    Args:
    Compares the current Transformers version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _transformers_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)


def is_accelerate_version(operation: str, version: str):
    """
    Args:
    Compares the current Accelerate version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _accelerate_available:
        return False
    return compare_versions(parse(_accelerate_version), operation, version)


def is_k_diffusion_version(operation: str, version: str):
    """
    Args:
    Compares the current k-diffusion version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _k_diffusion_available:
        return False
    return compare_versions(parse(_k_diffusion_version), operation, version)


class OptionalDependencyNotAvailable(BaseException):
    """An error indicating that an optional dependency of Diffusers was not found in the environment."""
