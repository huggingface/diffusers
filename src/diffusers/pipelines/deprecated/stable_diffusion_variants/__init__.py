from typing import TYPE_CHECKING

from ....utils import (
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
    from ....utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_cycle_diffusion"] = ["CycleDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint_legacy"] = ["StableDiffusionInpaintPipelineLegacy"]
    _import_structure["pipeline_stable_diffusion_model_editing"] = ["StableDiffusionModelEditingPipeline"]

    _import_structure["pipeline_stable_diffusion_paradigms"] = ["StableDiffusionParadigmsPipeline"]
    _import_structure["pipeline_stable_diffusion_pix2pix_zero"] = ["StableDiffusionPix2PixZeroPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ....utils.dummy_torch_and_transformers_objects import *

    else:
        from .pipeline_cycle_diffusion import CycleDiffusionPipeline
        from .pipeline_stable_diffusion_inpaint_legacy import StableDiffusionInpaintPipelineLegacy
        from .pipeline_stable_diffusion_model_editing import StableDiffusionModelEditingPipeline
        from .pipeline_stable_diffusion_paradigms import StableDiffusionParadigmsPipeline
        from .pipeline_stable_diffusion_pix2pix_zero import StableDiffusionPix2PixZeroPipeline

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
