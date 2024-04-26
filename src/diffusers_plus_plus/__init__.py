__version__ = "0.28.0.dev0"

from typing import TYPE_CHECKING

from ..diffusers.utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
    is_transformers_available,
)


# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "models": [],
    "pipelines": [],
    "schedulers": [],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_flax_available",
        "is_inflect_available",
        "is_invisible_watermark_available",
        "is_k_diffusion_available",
        "is_k_diffusion_version",
        "is_librosa_available",
        "is_note_seq_available",
        "is_onnx_available",
        "is_scipy_available",
        "is_torch_available",
        "is_torchsde_available",
        "is_transformers_available",
        "is_transformers_version",
        "is_unidecode_available",
        "logging",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..diffusers.utils import dummy_pt_objects  # noqa F403

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

else:
    _import_structure["models"].extend(
        [
            "ELLAProxyUNet",
        ]
    )

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..diffusers.utils import dummy_torch_and_transformers_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_objects"] = [
        name for name in dir(dummy_torch_and_transformers_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(
        [
            "StableDiffusionPipeline",
        ]
    )


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..diffusers.utils.dummy_pt_objects import *  # noqa F403
    else:
        from .models import (
            ELLAProxyUNet,
        )
    
    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..diffusers.utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .pipelines_plus_plus import (
            StableDiffusionPipeline,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )

