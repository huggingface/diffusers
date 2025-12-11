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
    _import_structure["encoders"] = [
        "Flux2TextEncoderStep",
        "Flux2RemoteTextEncoderStep",
        "Flux2VaeEncoderStep",
    ]
    _import_structure["before_denoise"] = [
        "Flux2SetTimestepsStep",
        "Flux2PrepareLatentsStep",
        "Flux2RoPEInputsStep",
        "Flux2PrepareImageLatentsStep",
    ]
    _import_structure["denoise"] = [
        "Flux2LoopDenoiser",
        "Flux2LoopAfterDenoiser",
        "Flux2DenoiseLoopWrapper",
        "Flux2DenoiseStep",
    ]
    _import_structure["decoders"] = ["Flux2DecodeStep"]
    _import_structure["inputs"] = [
        "Flux2ProcessImagesInputStep",
        "Flux2TextInputStep",
    ]
    _import_structure["modular_blocks"] = [
        "ALL_BLOCKS",
        "AUTO_BLOCKS",
        "REMOTE_AUTO_BLOCKS",
        "TEXT2IMAGE_BLOCKS",
        "IMAGE_CONDITIONED_BLOCKS",
        "Flux2AutoBlocks",
        "Flux2AutoVaeEncoderStep",
        "Flux2BeforeDenoiseStep",
        "Flux2VaeEncoderSequentialStep",
    ]
    _import_structure["modular_pipeline"] = ["Flux2ModularPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .before_denoise import (
            Flux2PrepareImageLatentsStep,
            Flux2PrepareLatentsStep,
            Flux2RoPEInputsStep,
            Flux2SetTimestepsStep,
        )
        from .decoders import Flux2DecodeStep
        from .denoise import (
            Flux2DenoiseLoopWrapper,
            Flux2DenoiseStep,
            Flux2LoopAfterDenoiser,
            Flux2LoopDenoiser,
        )
        from .encoders import (
            Flux2RemoteTextEncoderStep,
            Flux2TextEncoderStep,
            Flux2VaeEncoderStep,
        )
        from .inputs import (
            Flux2ProcessImagesInputStep,
            Flux2TextInputStep,
        )
        from .modular_blocks import (
            ALL_BLOCKS,
            AUTO_BLOCKS,
            IMAGE_CONDITIONED_BLOCKS,
            REMOTE_AUTO_BLOCKS,
            TEXT2IMAGE_BLOCKS,
            Flux2AutoBlocks,
            Flux2AutoVaeEncoderStep,
            Flux2BeforeDenoiseStep,
            Flux2VaeEncoderSequentialStep,
        )
        from .modular_pipeline import Flux2ModularPipeline
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
