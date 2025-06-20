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
    _import_structure["modular_pipeline_presets"] = ["StableDiffusionXLAutoPipeline"]
    _import_structure["modular_loader"] = ["StableDiffusionXLModularLoader"]
    _import_structure["encoders"] = ["StableDiffusionXLAutoIPAdapterStep", "StableDiffusionXLTextEncoderStep", "StableDiffusionXLAutoVaeEncoderStep"]
    _import_structure["decoders"] = ["StableDiffusionXLAutoDecodeStep"]
    _import_structure["modular_block_mappings"] = ["TEXT2IMAGE_BLOCKS", "IMAGE2IMAGE_BLOCKS", "INPAINT_BLOCKS", "CONTROLNET_BLOCKS", "CONTROLNET_UNION_BLOCKS", "IP_ADAPTER_BLOCKS", "AUTO_BLOCKS", "SDXL_SUPPORTED_BLOCKS"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modular_pipeline_presets import StableDiffusionXLAutoPipeline
        from .modular_loader import StableDiffusionXLModularLoader
        from .encoders import StableDiffusionXLAutoIPAdapterStep, StableDiffusionXLTextEncoderStep, StableDiffusionXLAutoVaeEncoderStep
        from .decoders import StableDiffusionXLAutoDecodeStep
        from .modular_block_mappings import SDXL_SUPPORTED_BLOCKS, TEXT2IMAGE_BLOCKS, IMAGE2IMAGE_BLOCKS, INPAINT_BLOCKS, CONTROLNET_BLOCKS, CONTROLNET_UNION_BLOCKS, IP_ADAPTER_BLOCKS, AUTO_BLOCKS
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
