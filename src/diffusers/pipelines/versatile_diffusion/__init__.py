from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .pipeline_versatile_diffusion import VersatileDiffusionPipeline
    from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
    from .pipeline_versatile_diffusion_text_image import VersatileDiffusionTextToImagePipeline
    from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline
