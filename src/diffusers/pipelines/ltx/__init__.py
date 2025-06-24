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
    _import_structure["modeling_latent_upsampler"] = ["LTXLatentUpsamplerModel"]
    _import_structure["pipeline_ltx"] = ["LTXPipeline"]
    _import_structure["pipeline_ltx_condition"] = ["LTXConditionPipeline"]
    _import_structure["pipeline_ltx_image2video"] = ["LTXImageToVideoPipeline"]
    _import_structure["pipeline_ltx_latent_upsample"] = ["LTXLatentUpsamplePipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .modeling_latent_upsampler import LTXLatentUpsamplerModel
        from .pipeline_ltx import LTXPipeline
        from .pipeline_ltx_condition import LTXConditionPipeline
        from .pipeline_ltx_image2video import LTXImageToVideoPipeline
        from .pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline

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
