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
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["modeling_wuerstchen3_common"] = ["WuerstchenV3Unet"]
    _import_structure["pipeline_wuerstchen3"] = ["WuerstchenV3DecoderPipeline"]
    _import_structure["pipeline_wuerstchen3_combined"] = ["WuerstchenV3CombinedPipeline"]
    _import_structure["pipeline_wuerstchen3_prior"] = ["WuerstchenV3PriorPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modeling_wuerstchen3_common import WuerstchenV3Unet
        from .pipeline_wuerstchen3 import WuerstchenV3DecoderPipeline
        from .pipeline_wuerstchen3_combined import WuerstchenV3CombinedPipeline
        from .pipeline_wuerstchen3_prior import WuerstchenV3PriorPipeline
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
