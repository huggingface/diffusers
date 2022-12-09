from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


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
    from .modeling_text_unet import UNetFlatConditionModel
    from .pipeline_versatile_diffusion import VersatileDiffusionPipeline
    from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
    from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
    from .pipeline_versatile_diffusion_text_to_image import VersatileDiffusionTextToImagePipeline
