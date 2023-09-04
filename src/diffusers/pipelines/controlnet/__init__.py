from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}
_dummy_objects = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()

except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))


else:
    _import_structure["multicontrolnet"] = ["MultiControlNetModel"]
    _import_structure["pipeline_controlnet"] = ["StableDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_img2img"] = ["StableDiffusionControlNetImg2ImgPipeline"]
    _import_structure["pipeline_controlnet_inpaint"] = ["StableDiffusionControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_sd_xl"] = ["StableDiffusionXLControlNetPipeline"]
    _import_structure["pipeline_controlnet_sd_xl_img2img"] = ["StableDiffusionXLControlNetImg2ImgPipeline"]
    _import_structure["pipeline_controlnet_inpaint_sd_xl"] = ["StableDiffusionXLControlNetInpaintPipeline"]

try:
    if not (is_transformers_available() and is_flax_available()):
        raise OptionalDependencyNotAvailable()

except OptionalDependencyNotAvailable:
    from ...utils import dummy_flax_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))

else:
    _import_structure["pipeline_flax_controlnet"] = ["FlaxStableDiffusionControlNetPipeline"]

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)

for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
