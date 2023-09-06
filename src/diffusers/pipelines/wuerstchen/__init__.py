from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .modeling_paella_vq_model import PaellaVQModel
    from .modeling_wuerstchen_diffnext import WuerstchenDiffNeXt
    from .modeling_wuerstchen_prior import WuerstchenPrior
    from .pipeline_wuerstchen import WuerstchenDecoderPipeline
    from .pipeline_wuerstchen_combined import WuerstchenCombinedPipeline
    from .pipeline_wuerstchen_prior import WuerstchenPriorPipeline
