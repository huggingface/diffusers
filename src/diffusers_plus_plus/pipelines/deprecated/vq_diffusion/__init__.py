from typing import TYPE_CHECKING

from ....utils import (
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
    from ....utils.dummy_torch_and_transformers_objects import (
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


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ....utils.dummy_torch_and_transformers_objects import (
            LearnedClassifierFreeSamplingEmbeddings,
            VQDiffusionPipeline,
        )
    else:
        from .pipeline_vq_diffusion import LearnedClassifierFreeSamplingEmbeddings, VQDiffusionPipeline

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
