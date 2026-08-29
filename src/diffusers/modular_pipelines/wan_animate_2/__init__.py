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
    _import_structure["modular_blocks_wan_animate_2"] = ["WanAnimate2Blocks"]
    _import_structure["modular_blocks_wan_animate_2_distilled"] = ["WanAnimate2DistilledBlocks"]
    _import_structure["modular_pipeline"] = [
        "WanAnimate2DistilledModularPipeline",
        "WanAnimate2ModularPipeline",
    ]
    _import_structure["video_processor"] = ["WanAnimate2VideoProcessor"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modular_blocks_wan_animate_2 import WanAnimate2Blocks
        from .modular_blocks_wan_animate_2_distilled import WanAnimate2DistilledBlocks
        from .modular_pipeline import (
            WanAnimate2DistilledModularPipeline,
            WanAnimate2ModularPipeline,
        )
        from .video_processor import WanAnimate2VideoProcessor
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
