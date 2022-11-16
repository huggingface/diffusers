from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .pipeline_vq_diffusion import LearnedClassifierFreeSamplingEmbeddings, VQDiffusionPipeline
