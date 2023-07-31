from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
)


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import *
else:
    from .pipeline_kandinsky2_2 import KandinskyV22Pipeline
    from .pipeline_kandinsky2_2_combined import (
        KandinskyV22CombinedPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22InpaintCombinedPipeline,
    )
    from .pipeline_kandinsky2_2_controlnet import KandinskyV22ControlnetPipeline
    from .pipeline_kandinsky2_2_controlnet_img2img import KandinskyV22ControlnetImg2ImgPipeline
    from .pipeline_kandinsky2_2_img2img import KandinskyV22Img2ImgPipeline
    from .pipeline_kandinsky2_2_inpainting import KandinskyV22InpaintPipeline
    from .pipeline_kandinsky2_2_prior import KandinskyV22PriorPipeline
    from .pipeline_kandinsky2_2_prior_emb2emb import KandinskyV22PriorEmb2EmbPipeline
