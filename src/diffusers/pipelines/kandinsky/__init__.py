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
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_kandinsky"] = ["KandinskyPipeline"]
    _import_structure["pipeline_kandinsky_combined"] = [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyInpaintCombinedPipeline",
    ]
    _import_structure["pipeline_kandinsky_img2img"] = ["KandinskyImg2ImgPipeline"]
    _import_structure["pipeline_kandinsky_inpaint"] = ["KandinskyInpaintPipeline"]
    _import_structure["pipeline_kandinsky_prior"] = ["KandinskyPriorPipeline", "KandinskyPriorPipelineOutput"]
    _import_structure["text_encoder"] = ["MultilingualCLIP"]


if TYPE_CHECKING:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        from .pipeline_kandinsky import KandinskyPipeline
        from .pipeline_kandinsky_combined import (
            KandinskyCombinedPipeline,
            KandinskyImg2ImgCombinedPipeline,
            KandinskyInpaintCombinedPipeline,
        )
        from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline
        from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
        from .pipeline_kandinsky_prior import KandinskyPriorPipeline, KandinskyPriorPipelineOutput
        from .text_encoder import MultilingualCLIP

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
