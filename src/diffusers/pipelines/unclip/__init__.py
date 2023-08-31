from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_import_structure = {}
_dummy_objects = {}


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import UnCLIPImageVariationPipeline, UnCLIPPipeline

    _dummy_objects.update(
        {"UnCLIPImageVariationPipeline": UnCLIPImageVariationPipeline, "UnCLIPPipeline": UnCLIPPipeline}
    )
else:
    _import_structure["pipeline_unclip"] = ["UnCLIPPipeline"]
    _import_structure["pipeline_unclip_image_variation"] = ["UnCLIPImageVariationPipeline"]
    _import_structure["text_proj"] = ["UnCLIPTextProjModel"]

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
