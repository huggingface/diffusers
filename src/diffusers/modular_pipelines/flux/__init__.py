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
    _import_structure["encoders"] = ["FluxTextEncoderStep"]
    _import_structure["modular_blocks"] = [
        "ALL_BLOCKS",
        "AUTO_BLOCKS",
        "AUTO_BLOCKS_KONTEXT",
        "FLUX_KONTEXT_BLOCKS",
        "TEXT2IMAGE_BLOCKS",
        "FluxAutoBeforeDenoiseStep",
        "FluxAutoBlocks",
        "FluxAutoDecodeStep",
        "FluxAutoDenoiseStep",
        "FluxKontextAutoBlocks",
        "FluxKontextAutoDenoiseStep",
        "FluxKontextBeforeDenoiseStep",
    ]
    _import_structure["modular_pipeline"] = ["FluxKontextModularPipeline", "FluxModularPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .encoders import FluxTextEncoderStep
        from .modular_blocks import (
            ALL_BLOCKS,
            AUTO_BLOCKS,
            AUTO_BLOCKS_KONTEXT,
            FLUX_KONTEXT_BLOCKS,
            TEXT2IMAGE_BLOCKS,
            FluxAutoBeforeDenoiseStep,
            FluxAutoBlocks,
            FluxAutoDecodeStep,
            FluxAutoDenoiseStep,
            FluxKontextAutoBlocks,
            FluxKontextAutoDenoiseStep,
            FluxKontextBeforeDenoiseStep,
        )
        from .modular_pipeline import FluxKontextModularPipeline, FluxModularPipeline
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
