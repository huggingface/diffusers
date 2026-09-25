"""Kandinsky 6 pipelines."""

from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_pt_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["pipeline_kandinsky6_ti2va"] = ["Kandinsky6TI2VAPipeline"]
    _import_structure["pipeline_kandinsky6_sr"] = ["Kandinsky6SRPipeline"]
    _import_structure["latent_upscaler"] = ["Kandinsky6SRLatentUpscalerBank"]
    _import_structure["pipeline_output"] = ["Kandinsky6SRPipelineOutput", "Kandinsky6TI2VAPipelineOutput"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_pt_objects import *
    else:
        from .pipeline_kandinsky6_ti2va import Kandinsky6TI2VAPipeline
        from .pipeline_kandinsky6_sr import Kandinsky6SRPipeline
        from .latent_upscaler import Kandinsky6SRLatentUpscalerBank
        from .pipeline_output import Kandinsky6SRPipelineOutput, Kandinsky6TI2VAPipelineOutput
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


__all__ = [
    "Kandinsky6TI2VAPipeline",
    "Kandinsky6TI2VAPipelineOutput",
    "Kandinsky6SRPipeline",
    "Kandinsky6SRLatentUpscalerBank",
    "Kandinsky6SRPipelineOutput",
]
