from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["StableDiffusionXLPipelineOutput"]}

if is_transformers_available() and is_flax_available():
    _import_structure["pipeline_output"].extend(["FlaxStableDiffusionXLPipelineOutput"])
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_stable_diffusion_xl"] = ["StableDiffusionXLPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_img2img"] = ["StableDiffusionXLImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_inpaint"] = ["StableDiffusionXLInpaintPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_instruct_pix2pix"] = ["StableDiffusionXLInstructPix2PixPipeline"]

if is_transformers_available() and is_flax_available():
    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState

    _additional_imports.update({"PNDMSchedulerState": PNDMSchedulerState})
    _import_structure["pipeline_flax_stable_diffusion_xl"] = ["FlaxStableDiffusionXLPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
        from .pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
        from .pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
        from .pipeline_stable_diffusion_xl_instruct_pix2pix import StableDiffusionXLInstructPix2PixPipeline

    try:
        if not (is_transformers_available() and is_flax_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_flax_objects import *
    else:
        from .pipeline_flax_stable_diffusion_xl import (
            FlaxStableDiffusionXLPipeline,
        )
        from .pipeline_output import FlaxStableDiffusionXLPipelineOutput

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
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
