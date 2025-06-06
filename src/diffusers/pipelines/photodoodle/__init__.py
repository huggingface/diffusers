"""
PhotoDoodle pipeline for image generation.
"""

from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
)

_dummy_objects = {}
_import_structure = {
    "pipeline_photodoodle": ["PhotoDoodlePipeline"],
}

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipeline_photodoodle import PhotoDoodlePipeline

if TYPE_CHECKING:
    from .pipeline_photodoodle import PhotoDoodlePipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    ) 