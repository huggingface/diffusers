from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_opencv_available,
    is_sentencepiece_available,
    is_torch_available,
    is_torch_npu_available,
    is_transformers_available,
    is_transformers_version,
)


# These modules contain pipelines from multiple libraries/frameworks
_dummy_objects = {}
_import_structure = {
    "latent_diffusion": [],
    "stable_diffusion": [],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["ddpm"] = ["DDPMPipeline"]
    _import_structure["latent_diffusion"].extend(["LDMSuperResolutionPipeline"])
    _import_structure["pipeline_utils"] = [
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ]
try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["latent_diffusion"].extend(["LDMTextToImagePipeline"])
    _import_structure["stable_diffusion"].extend(
        [
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionPipeline",
        ]
    )


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *  # noqa F403

    else:
        from .ddpm import DDPMPipeline
        from .latent_diffusion import LDMSuperResolutionPipeline
        from .pipeline_utils import (
            DiffusionPipeline,
            ImagePipelineOutput,
        )

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_torch_and_transformers_objects import *
    else:
        from .latent_diffusion import LDMTextToImagePipeline
        from .stable_diffusion import (
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionPipeline,
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
