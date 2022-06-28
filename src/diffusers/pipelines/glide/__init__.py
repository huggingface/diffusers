from ...utils import is_transformers_available


if is_transformers_available():
    from .pipeline_glide import CLIPTextModel, GlidePipeline
