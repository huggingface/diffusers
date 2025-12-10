from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_kandinsky"] = ["Kandinsky5T2VPipeline"]
    _import_structure["pipeline_kandinsky_i2i"] = ["Kandinsky5I2IPipeline"]
    _import_structure["pipeline_kandinsky_i2v"] = ["Kandinsky5I2VPipeline"]
    _import_structure["pipeline_kandinsky_t2i"] = ["Kandinsky5T2IPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_kandinsky import Kandinsky5T2VPipeline
        from .pipeline_kandinsky_i2i import Kandinsky5I2IPipeline
        from .pipeline_kandinsky_i2v import Kandinsky5I2VPipeline
        from .pipeline_kandinsky_t2i import Kandinsky5T2IPipeline

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
