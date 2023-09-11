from typing import TYPE_CHECKING

from ...utils import (
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
        ImageTextPipelineOutput,
        UniDiffuserPipeline,
    )

    _dummy_objects.update(
        {"ImageTextPipelineOutput": ImageTextPipelineOutput, "UniDiffuserPipeline": UniDiffuserPipeline}
    )
else:
    _import_structure["modeling_text_decoder"] = ["UniDiffuserTextDecoder"]
    _import_structure["modeling_uvit"] = ["UniDiffuserModel", "UTransformer2DModel"]
    _import_structure["pipeline_unidiffuser"] = ["ImageTextPipelineOutput", "UniDiffuserPipeline"]


if TYPE_CHECKING:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            ImageTextPipelineOutput,
            UniDiffuserPipeline,
        )
    else:
        from .modeling_text_decoder import UniDiffuserTextDecoder
        from .modeling_uvit import UniDiffuserModel, UTransformer2DModel
        from .pipeline_unidiffuser import ImageTextPipelineOutput, UniDiffuserPipeline

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
