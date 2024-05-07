from typing import TYPE_CHECKING

from ....utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}

_import_structure = {
    "oms_pipeline": [],
}


try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ....utils import dummy_torch_and_transformers_objects


    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["oms_pipeline"].extend(
        [
            "OmsDiffusionPipeline",
            "ClothAdapter"
        ]
    )

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ....utils.dummy_torch_and_transformers_objects import *
    else:
        from .oms_pipeline import ClothAdapter, OmsDiffusionPipeline


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


