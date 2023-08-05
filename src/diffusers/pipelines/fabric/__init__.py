from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
)

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_fabric import FabricPipeline

