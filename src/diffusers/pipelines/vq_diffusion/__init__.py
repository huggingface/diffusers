from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}
_dummy_objects = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        LearnedClassifierFreeSamplingEmbeddings,
        VQDiffusionPipeline,
    )

    _dummy_objects.update(
        {
            "LearnedClassifierFreeSamplingEmbeddings": LearnedClassifierFreeSamplingEmbeddings,
            "VQDiffusionPipeline": VQDiffusionPipeline,
        }
    )
else:
    _import_structure["pipeline_vq_diffusion"] = ["LearnedClassifierFreeSamplingEmbeddings", "VQDiffusionPipeline"]

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)
