from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import UnCLIPImageVariationPipeline, UnCLIPPipeline
else:
    from .pipeline_unclip import UnCLIPPipeline
    from .pipeline_unclip_image_variation import UnCLIPImageVariationPipeline
    from .text_proj import UnCLIPTextProjModel
