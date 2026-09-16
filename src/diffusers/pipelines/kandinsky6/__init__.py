"""Kandinsky 6 pipelines."""

from typing import TYPE_CHECKING

from ...utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, is_torch_available


_import_structure = {}

if is_torch_available():
    _import_structure["pipeline_kandinsky6_ti2va"] = ["Kandinsky6TI2VAPipeline"]
    _import_structure["pipeline_kandinsky6_sr"] = ["Kandinsky6SRPipeline"]
    _import_structure["latent_upscaler"] = ["Kandinsky6SRLatentUpscalerBank"]
    _import_structure["pipeline_output"] = ["Kandinsky6SRPipelineOutput", "Kandinsky6TI2VAPipelineOutput"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
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


__all__ = [
    "Kandinsky6TI2VAPipeline",
    "Kandinsky6TI2VAPipelineOutput",
    "Kandinsky6SRPipeline",
    "Kandinsky6SRLatentUpscalerBank",
    "Kandinsky6SRPipelineOutput",
]
