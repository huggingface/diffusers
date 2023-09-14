from typing import TYPE_CHECKING

from ...utils import (
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
    _import_structure["modeling_paella_vq_model"] = ["PaellaVQModel"]
    _import_structure["modeling_wuerstchen_diffnext"] = ["WuerstchenDiffNeXt"]
    _import_structure["modeling_wuerstchen_prior"] = ["WuerstchenPrior"]
    _import_structure["pipeline_wuerstchen"] = ["WuerstchenDecoderPipeline"]
    _import_structure["pipeline_wuerstchen_combined"] = ["WuerstchenCombinedPipeline"]
    _import_structure["pipeline_wuerstchen_prior"] = ["DEFAULT_STAGE_C_TIMESTEPS", "WuerstchenPriorPipeline"]


if TYPE_CHECKING:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modeling_paella_vq_model import PaellaVQModel
        from .modeling_wuerstchen_diffnext import WuerstchenDiffNeXt
        from .modeling_wuerstchen_prior import WuerstchenPrior
        from .pipeline_wuerstchen import WuerstchenDecoderPipeline
        from .pipeline_wuerstchen_combined import WuerstchenCombinedPipeline
        from .pipeline_wuerstchen_prior import WuerstchenPriorPipeline
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
