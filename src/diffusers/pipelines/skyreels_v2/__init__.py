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
    _import_structure["pipeline_skyreels_v2"] = ["SkyReelsV2Pipeline"]
    _import_structure["pipeline_skyreels_v2_diffusion_forcing"] = ["SkyReelsV2DiffusionForcingPipeline"]
    _import_structure["pipeline_skyreels_v2_diffusion_forcing_i2v"] = [
        "SkyReelsV2DiffusionForcingImageToVideoPipeline"
    ]
    _import_structure["pipeline_skyreels_v2_diffusion_forcing_v2v"] = [
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline"
    ]
    _import_structure["pipeline_skyreels_v2_i2v"] = ["SkyReelsV2ImageToVideoPipeline"]
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_skyreels_v2 import SkyReelsV2Pipeline
        from .pipeline_skyreels_v2_diffusion_forcing import SkyReelsV2DiffusionForcingPipeline
        from .pipeline_skyreels_v2_diffusion_forcing_i2v import SkyReelsV2DiffusionForcingImageToVideoPipeline
        from .pipeline_skyreels_v2_diffusion_forcing_v2v import SkyReelsV2DiffusionForcingVideoToVideoPipeline
        from .pipeline_skyreels_v2_i2v import SkyReelsV2ImageToVideoPipeline

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
