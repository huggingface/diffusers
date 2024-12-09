from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        MeissonicImg2ImgPipeline,
        MeissonicInpaintPipeline,
        MeissonicPipeline,
    )

    _dummy_objects.update(
        {
            "MeissonicPipeline": MeissonicPipeline,
            "MeissonicImg2ImgPipeline": MeissonicImg2ImgPipeline,
            "MeissonicInpaintPipeline": MeissonicInpaintPipeline,
        }
    )
else:
    _import_structure["pipeline_meissonic"] = ["MeissonicPipeline"]
    _import_structure["pipeline_meissonic_img2img"] = ["MeissonicImg2ImgPipeline"]
    _import_structure["pipeline_meissonic_inpaint"] = ["MeissonicInpaintPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            MeissonicPipeline,
        )
    else:
        from .pipeline_meissonic import MeissonicPipeline
        from .pipeline_meissonic_img2img import MeissonicImg2ImgPipeline
        from .pipeline_meissonic_inpaint import MeissonicInpaintPipeline

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
