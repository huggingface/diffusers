from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
)


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_stable_diffusion_adapter import StableDiffusionAdapterPipeline
    from .pipeline_stable_diffusion_xl_adapter import StableDiffusionXLAdapterPipeline
