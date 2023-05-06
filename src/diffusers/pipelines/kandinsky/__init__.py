from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
    is_multilingual_clip_available,
)


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")) and is_multilingual_clip_available:
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_and_multilingual_clip_diffusion_objects import KandinskyPipeline
else:
    from .pipeline_kandinsky import KandinskyPipeline
    from .text_proj import KandinskyTextProjModel
