from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .modules import DiffNeXt, EfficientNetEncoder, Prior
    from .pipeline_wuerstchen import WuerstchenGeneratorPipeline
    from .pipeline_wuerstchen_prior import WuerstchenPriorPipeline
