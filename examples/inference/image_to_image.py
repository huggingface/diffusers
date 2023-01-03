import warnings

from diffusers import StableDiffusionImg2ImgPipeline  # noqa F401


warnings.warn(
    "The `image_to_image.py` script is outdated. Please use directly `from diffusers import"
    " StableDiffusionImg2ImgPipeline` instead."
)
