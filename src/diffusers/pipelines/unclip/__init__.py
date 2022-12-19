from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .pipeline_unclip import UnCLIPPipeline
    from .text_proj import UnCLIPTextProjModel
