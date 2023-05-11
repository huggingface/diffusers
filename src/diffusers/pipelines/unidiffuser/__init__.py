from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


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
