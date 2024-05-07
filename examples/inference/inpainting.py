import warnings

from diffusers import (
    StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline,
)


warnings.warn(
    "The `inpainting.py` script is outdated. Please use directly `from diffusers import"
    " StableDiffusionInpaintPipeline` instead."
)
