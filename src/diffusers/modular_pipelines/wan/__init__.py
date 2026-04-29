from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["modular_blocks_wan"] = ["WanBlocks"]
    _import_structure["modular_blocks_wan22"] = ["Wan22Blocks"]
    _import_structure["modular_blocks_wan22_i2v"] = ["Wan22Image2VideoBlocks"]
    _import_structure["modular_blocks_wan_i2v"] = ["WanImage2VideoAutoBlocks"]
    _import_structure["modular_pipeline"] = [
        "Wan22Image2VideoModularPipeline",
        "Wan22ModularPipeline",
        "WanImage2VideoModularPipeline",
        "WanModularPipeline",
    ]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modular_blocks_wan import WanBlocks
        from .modular_blocks_wan22 import Wan22Blocks
        from .modular_blocks_wan22_i2v import Wan22Image2VideoBlocks
        from .modular_blocks_wan_i2v import WanImage2VideoAutoBlocks
        from .modular_pipeline import (
            Wan22Image2VideoModularPipeline,
            Wan22ModularPipeline,
            WanImage2VideoModularPipeline,
            WanModularPipeline,
        )
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
