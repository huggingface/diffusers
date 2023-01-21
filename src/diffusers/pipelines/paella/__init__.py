from ...utils import is_torch_available


if is_torch_available():
    from .pipeline_paella import PaellaTextToImagePipeline
