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

from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    is_flax_available,
    is_inflect_available,
    is_modelcards_available,
    is_onnx_available,
    is_scipy_available,
    is_tf_available,
    is_torch_available,
    is_transformers_available,
    is_unidecode_available,
    requires_backends,
)
from .logging import get_logger
from .outputs import BaseOutput


logger = get_logger(__name__)


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "diffusers")


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
DIFFUSERS_CACHE = default_cache_path
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
<<<<<<< HEAD
=======


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


_modelcards_available = importlib.util.find_spec("modelcards") is not None
try:
    _modelcards_version = importlib_metadata.version("modelcards")
    logger.debug(f"Successfully imported modelcards version {_modelcards_version}")
except importlib_metadata.PackageNotFoundError:
    _modelcards_available = False

_torch_scatter_available = importlib.util.find_spec("torch_scatter") is not None
try:
    _torch_scatter_version = importlib_metadata.version("torch_scatter")
    logger.debug(f"Successfully imported torch_scatter version {_torch_scatter_version}")
except importlib_metadata.PackageNotFoundError:
    _torch_scatter_available = False

_torch_scatter_available = importlib.util.find_spec("torch_geometric") is not None
try:
    _torch_geometric_version = importlib_metadata.version("torch_geometric")
    logger.debug(f"Successfully imported torch_geometric version {_torch_geometric_version}")
except importlib_metadata.PackageNotFoundError:
    _torch_geometric_available = False


def is_transformers_available():
    return _transformers_available


def is_inflect_available():
    return _inflect_available


def is_unidecode_available():
    return _unidecode_available


def is_modelcards_available():
    return _modelcards_available


def is_torch_scatter_available():
    return _torch_scatter_available


def is_torch_geometric_available():
    # the model source of the Molecule Generation GNN requires a specific torch geometric version
    # for more info, see the original repo https://github.com/MinkaiXu/GeoDiff or our colab in readme
    return _torch_geometric_version == "1.7.2"


class RepositoryNotFoundError(HTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or with a private repo name the user does
    not have access to.
    """


class EntryNotFoundError(HTTPError):
    """Raised when trying to access a hf.co URL with a valid repository and revision but an invalid filename."""


class RevisionNotFoundError(HTTPError):
    """Raised when trying to access a hf.co URL with a valid repository but an invalid revision."""


TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""


UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""


INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

TORCH_GEOMETRIC_IMPORT_ERROR = """
{0} requires version 1.7.2 of torch_geometric but it was not found in your environment. You can install it with conda:
`conda install -c rusty1s pytorch-geometric=1.7.2`, given pytorch 1.8
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("transformers", (is_transformers_available, TRANSFORMERS_IMPORT_ERROR)),
        ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
        ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
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


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)
>>>>>>> bf87817 (add checking for imports)
