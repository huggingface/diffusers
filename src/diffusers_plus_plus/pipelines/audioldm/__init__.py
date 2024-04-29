from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        AudioLDMPipeline,
    )

    _dummy_objects.update({"AudioLDMPipeline": AudioLDMPipeline})
else:
    _import_structure["pipeline_audioldm"] = ["AudioLDMPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            AudioLDMPipeline,
        )

    else:
        from .pipeline_audioldm import AudioLDMPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
