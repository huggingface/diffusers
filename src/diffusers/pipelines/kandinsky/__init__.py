from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import KandinskyPipeline, KandinskyPriorPipeline
else:
    from .pipeline_kandinsky import KandinskyPipeline
    from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline
    from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
    from .pipeline_kandinsky_prior import KandinskyPriorPipeline
    from .text_encoder import MultilingualCLIP
