import warnings

from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline  # noqa F401


warnings.warn(
    "The `inpainting.py` script is outdated. Please use directly `from diffusers import"
    " StableDiffusionInpaintPipeline` instead."
)
