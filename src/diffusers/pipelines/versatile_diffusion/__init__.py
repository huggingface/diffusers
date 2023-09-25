from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )

    _dummy_objects.update(
        {
            "VersatileDiffusionDualGuidedPipeline": VersatileDiffusionDualGuidedPipeline,
            "VersatileDiffusionImageVariationPipeline": VersatileDiffusionImageVariationPipeline,
            "VersatileDiffusionPipeline": VersatileDiffusionPipeline,
            "VersatileDiffusionTextToImagePipeline": VersatileDiffusionTextToImagePipeline,
        }
    )
else:
    _import_structure["modeling_text_unet"] = ["UNetFlatConditionModel"]
    _import_structure["pipeline_versatile_diffusion"] = ["VersatileDiffusionPipeline"]
    _import_structure["pipeline_versatile_diffusion_dual_guided"] = ["VersatileDiffusionDualGuidedPipeline"]
    _import_structure["pipeline_versatile_diffusion_image_variation"] = ["VersatileDiffusionImageVariationPipeline"]
    _import_structure["pipeline_versatile_diffusion_text_to_image"] = ["VersatileDiffusionTextToImagePipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
        )
    else:
        from .pipeline_versatile_diffusion import VersatileDiffusionPipeline
        from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
        from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
        from .pipeline_versatile_diffusion_text_to_image import VersatileDiffusionTextToImagePipeline

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
