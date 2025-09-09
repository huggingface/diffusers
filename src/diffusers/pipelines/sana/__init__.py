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
    _import_structure["pipeline_sana"] = ["SanaPipeline"]
    _import_structure["pipeline_sana_controlnet"] = ["SanaControlNetPipeline"]
    _import_structure["pipeline_sana_sprint"] = ["SanaSprintPipeline"]
    _import_structure["pipeline_sana_sprint_img2img"] = ["SanaSprintImg2ImgPipeline"]
    _import_structure["pipeline_sana_video"] = ["SanaVideoPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_sana import SanaPipeline
        from .pipeline_sana_controlnet import SanaControlNetPipeline
        from .pipeline_sana_sprint import SanaSprintPipeline
        from .pipeline_sana_sprint_img2img import SanaSprintImg2ImgPipeline
        from .pipeline_sana_video import SanaVideoPipeline
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
