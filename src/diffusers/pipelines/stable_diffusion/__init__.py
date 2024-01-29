from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["StableDiffusionPipelineOutput"]}

if is_transformers_available() and is_flax_available():
    _import_structure["pipeline_output"].extend(["FlaxStableDiffusionPipelineOutput"])
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["clip_image_project_model"] = ["CLIPImageProjection"]
    _import_structure["pipeline_cycle_diffusion"] = ["CycleDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion"] = ["StableDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion_attend_and_excite"] = ["StableDiffusionAttendAndExcitePipeline"]
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableDiffusionGLIGENPipeline"]
    _import_structure["pipeline_stable_diffusion_gligen_text_image"] = ["StableDiffusionGLIGENTextImagePipeline"]
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableDiffusionInpaintPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint_legacy"] = ["StableDiffusionInpaintPipelineLegacy"]
    _import_structure["pipeline_stable_diffusion_instruct_pix2pix"] = ["StableDiffusionInstructPix2PixPipeline"]
    _import_structure["pipeline_stable_diffusion_latent_upscale"] = ["StableDiffusionLatentUpscalePipeline"]
    _import_structure["pipeline_stable_diffusion_model_editing"] = ["StableDiffusionModelEditingPipeline"]
    _import_structure["pipeline_stable_diffusion_paradigms"] = ["StableDiffusionParadigmsPipeline"]
    _import_structure["pipeline_stable_diffusion_upscale"] = ["StableDiffusionUpscalePipeline"]
    _import_structure["pipeline_stable_unclip"] = ["StableUnCLIPPipeline"]
    _import_structure["pipeline_stable_unclip_img2img"] = ["StableUnCLIPImg2ImgPipeline"]
    _import_structure["safety_checker"] = ["StableDiffusionSafetyChecker"]
    _import_structure["stable_unclip_image_normalizer"] = ["StableUnCLIPImageNormalizer"]
try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        StableDiffusionImageVariationPipeline,
    )

    _dummy_objects.update({"StableDiffusionImageVariationPipeline": StableDiffusionImageVariationPipeline})
else:
    _import_structure["pipeline_stable_diffusion_image_variation"] = ["StableDiffusionImageVariationPipeline"]
try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        StableDiffusionDepth2ImgPipeline,
    )

    _dummy_objects.update(
        {
            "StableDiffusionDepth2ImgPipeline": StableDiffusionDepth2ImgPipeline,
        }
    )
else:
    _import_structure["pipeline_stable_diffusion_depth2img"] = ["StableDiffusionDepth2ImgPipeline"]

try:
    if not (is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_onnx_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_onnx_objects))
else:
    _import_structure["pipeline_onnx_stable_diffusion"] = [
        "OnnxStableDiffusionPipeline",
        "StableDiffusionOnnxPipeline",
    ]
    _import_structure["pipeline_onnx_stable_diffusion_img2img"] = ["OnnxStableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_onnx_stable_diffusion_inpaint"] = ["OnnxStableDiffusionInpaintPipeline"]
    _import_structure["pipeline_onnx_stable_diffusion_inpaint_legacy"] = ["OnnxStableDiffusionInpaintPipelineLegacy"]
    _import_structure["pipeline_onnx_stable_diffusion_upscale"] = ["OnnxStableDiffusionUpscalePipeline"]

if is_transformers_available() and is_flax_available():
    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState

    _additional_imports.update({"PNDMSchedulerState": PNDMSchedulerState})
    _import_structure["pipeline_flax_stable_diffusion"] = ["FlaxStableDiffusionPipeline"]
    _import_structure["pipeline_flax_stable_diffusion_img2img"] = ["FlaxStableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_flax_stable_diffusion_inpaint"] = ["FlaxStableDiffusionInpaintPipeline"]
    _import_structure["safety_checker_flax"] = ["FlaxStableDiffusionSafetyChecker"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        from .clip_image_project_model import CLIPImageProjection
        from .pipeline_stable_diffusion import (
            StableDiffusionPipeline,
            StableDiffusionPipelineOutput,
            StableDiffusionSafetyChecker,
        )
        from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
        from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        from .pipeline_stable_diffusion_instruct_pix2pix import (
            StableDiffusionInstructPix2PixPipeline,
        )
        from .pipeline_stable_diffusion_latent_upscale import (
            StableDiffusionLatentUpscalePipeline,
        )
        from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
        from .pipeline_stable_unclip import StableUnCLIPPipeline
        from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
        from .safety_checker import StableDiffusionSafetyChecker
        from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            StableDiffusionImageVariationPipeline,
        )
    else:
        from .pipeline_stable_diffusion_image_variation import (
            StableDiffusionImageVariationPipeline,
        )

    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import StableDiffusionDepth2ImgPipeline
    else:
        from .pipeline_stable_diffusion_depth2img import (
            StableDiffusionDepth2ImgPipeline,
        )

    try:
        if not (is_transformers_available() and is_onnx_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_onnx_objects import *
    else:
        from .pipeline_onnx_stable_diffusion import (
            OnnxStableDiffusionPipeline,
            StableDiffusionOnnxPipeline,
        )
        from .pipeline_onnx_stable_diffusion_img2img import (
            OnnxStableDiffusionImg2ImgPipeline,
        )
        from .pipeline_onnx_stable_diffusion_inpaint import (
            OnnxStableDiffusionInpaintPipeline,
        )
        from .pipeline_onnx_stable_diffusion_upscale import (
            OnnxStableDiffusionUpscalePipeline,
        )

    try:
        if not (is_transformers_available() and is_flax_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_flax_objects import *
    else:
        from .pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
        from .pipeline_flax_stable_diffusion_img2img import (
            FlaxStableDiffusionImg2ImgPipeline,
        )
        from .pipeline_flax_stable_diffusion_inpaint import (
            FlaxStableDiffusionInpaintPipeline,
        )
        from .pipeline_output import FlaxStableDiffusionPipelineOutput
        from .safety_checker_flax import FlaxStableDiffusionSafetyChecker

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
