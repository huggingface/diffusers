from ...utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_torch_available,
    is_transformers_available,
)


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .multicontrolnet import MultiControlNetModel
    from .pipeline_controlnet import StableDiffusionControlNetPipeline
    from .pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
    from .pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline


if is_transformers_available() and is_flax_available():
    from .pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline
