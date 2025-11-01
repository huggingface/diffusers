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
    _import_structure["pipeline_magi1"] = ["Magi1Pipeline"]
    _import_structure["pipeline_magi1_i2v"] = ["Magi1ImageToVideoPipeline"]
    _import_structure["pipeline_magi1_v2v"] = ["Magi1VideoToVideoPipeline"]
    _import_structure["pipeline_output"] = ["Magi1PipelineOutput"]
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_magi1 import Magi1Pipeline
        from .pipeline_magi1_i2v import Magi1ImageToVideoPipeline
        from .pipeline_magi1_v2v import Magi1VideoToVideoPipeline
        from .pipeline_output import Magi1PipelineOutput

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
