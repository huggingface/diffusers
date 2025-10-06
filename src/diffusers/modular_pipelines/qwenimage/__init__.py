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
    _import_structure["encoders"] = ["QwenImageTextEncoderStep"]
    _import_structure["modular_blocks"] = [
        "ALL_BLOCKS",
        "AUTO_BLOCKS",
        "CONTROLNET_BLOCKS",
        "EDIT_AUTO_BLOCKS",
        "EDIT_BLOCKS",
        "EDIT_INPAINT_BLOCKS",
        "EDIT_PLUS_AUTO_BLOCKS",
        "EDIT_PLUS_BLOCKS",
        "IMAGE2IMAGE_BLOCKS",
        "INPAINT_BLOCKS",
        "TEXT2IMAGE_BLOCKS",
        "QwenImageAutoBlocks",
        "QwenImageEditAutoBlocks",
        "QwenImageEditPlusAutoBlocks",
    ]
    _import_structure["modular_pipeline"] = [
        "QwenImageEditModularPipeline",
        "QwenImageEditPlusModularPipeline",
        "QwenImageModularPipeline",
    ]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .encoders import (
            QwenImageTextEncoderStep,
        )
        from .modular_blocks import (
            ALL_BLOCKS,
            AUTO_BLOCKS,
            CONTROLNET_BLOCKS,
            EDIT_AUTO_BLOCKS,
            EDIT_BLOCKS,
            EDIT_INPAINT_BLOCKS,
            EDIT_PLUS_AUTO_BLOCKS,
            EDIT_PLUS_BLOCKS,
            IMAGE2IMAGE_BLOCKS,
            INPAINT_BLOCKS,
            TEXT2IMAGE_BLOCKS,
            QwenImageAutoBlocks,
            QwenImageEditAutoBlocks,
            QwenImageEditPlusAutoBlocks,
        )
        from .modular_pipeline import (
            QwenImageEditModularPipeline,
            QwenImageEditPlusModularPipeline,
            QwenImageModularPipeline,
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
