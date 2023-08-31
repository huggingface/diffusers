from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}
_dummy_objects = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))

else:
    _import_structure["pipeline_output"] = ["AltDiffusionPipelineOutput"]
    _import_structure["modeling_roberta_series"] = ["RobertaSeriesModelWithTransformation"]
    _import_structure["pipeline_alt_diffusion"] = ["AltDiffusionPipeline"]
    _import_structure["pipeline_alt_diffusion_img2img"] = ["AltDiffusionImg2ImgPipeline"]

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
