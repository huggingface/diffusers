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
    _import_structure["encoders"] = ["StableDiffusionXLTextEncoderStep"]
    _import_structure["modular_blocks"] = [
        "ALL_BLOCKS",
        "AUTO_BLOCKS",
        "CONTROLNET_BLOCKS",
        "IMAGE2IMAGE_BLOCKS",
        "INPAINT_BLOCKS",
        "IP_ADAPTER_BLOCKS",
        "TEXT2IMAGE_BLOCKS",
        "StableDiffusionXLAutoBlocks",
        "StableDiffusionXLAutoControlnetStep",
        "StableDiffusionXLAutoDecodeStep",
        "StableDiffusionXLAutoIPAdapterStep",
        "StableDiffusionXLAutoVaeEncoderStep",
    ]
    _import_structure["modular_pipeline"] = ["StableDiffusionXLModularPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .encoders import (
            StableDiffusionXLTextEncoderStep,
        )
        from .modular_blocks import (
            ALL_BLOCKS,
            AUTO_BLOCKS,
            CONTROLNET_BLOCKS,
            IMAGE2IMAGE_BLOCKS,
            INPAINT_BLOCKS,
            IP_ADAPTER_BLOCKS,
            TEXT2IMAGE_BLOCKS,
            StableDiffusionXLAutoBlocks,
            StableDiffusionXLAutoControlnetStep,
            StableDiffusionXLAutoDecodeStep,
            StableDiffusionXLAutoIPAdapterStep,
            StableDiffusionXLAutoVaeEncoderStep,
        )
        from .modular_pipeline import StableDiffusionXLModularPipeline
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
