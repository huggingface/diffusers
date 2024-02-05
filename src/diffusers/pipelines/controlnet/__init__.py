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
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["multicontrolnet"] = ["MultiControlNetModel"]
    _import_structure["pipeline_controlnet"] = ["StableDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_blip_diffusion"] = ["BlipDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_img2img"] = ["StableDiffusionControlNetImg2ImgPipeline"]
    _import_structure["pipeline_controlnet_inpaint"] = ["StableDiffusionControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_inpaint_sd_xl"] = ["StableDiffusionXLControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_sd_xl"] = ["StableDiffusionXLControlNetPipeline"]
    _import_structure["pipeline_controlnet_sd_xl_img2img"] = ["StableDiffusionXLControlNetImg2ImgPipeline"]
try:
    if not (is_transformers_available() and is_flax_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_flax_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
else:
    _import_structure["pipeline_flax_controlnet"] = ["FlaxStableDiffusionControlNetPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .multicontrolnet import MultiControlNetModel
        from .pipeline_controlnet import StableDiffusionControlNetPipeline
        from .pipeline_controlnet_blip_diffusion import BlipDiffusionControlNetPipeline
        from .pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
        from .pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
        from .pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
        from .pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
        from .pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline

    try:
        if not (is_transformers_available() and is_flax_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_flax_and_transformers_objects import *  # noqa F403
    else:
        from .pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline


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
