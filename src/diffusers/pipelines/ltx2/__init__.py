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
    _import_structure["connectors"] = ["LTX2TextConnectors"]
    _import_structure["latent_upsampler"] = ["LTX2LatentUpsamplerModel"]
    _import_structure["pipeline_ltx2"] = ["LTX2Pipeline"]
    _import_structure["pipeline_ltx2_image2video"] = ["LTX2ImageToVideoPipeline"]
    _import_structure["pipeline_ltx2_latent_upsample"] = ["LTX2LatentUpsamplePipeline"]
    _import_structure["vocoder"] = ["LTX2Vocoder"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .connectors import LTX2TextConnectors
        from .latent_upsampler import LTX2LatentUpsamplerModel
        from .pipeline_ltx2 import LTX2Pipeline
        from .pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
        from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline
        from .vocoder import LTX2Vocoder

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
