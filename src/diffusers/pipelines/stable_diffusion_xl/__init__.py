from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}
_dummy_objects = {}

_import_structure["pipeline_output"] = ["StableDiffusionXLPipelineOutput"]

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

import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)

for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
