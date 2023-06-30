from ...utils import is_torch_available, is_transformers_available


if is_transformers_available() and is_torch_available():
    from .modules import Prior, DiffNeXt, EfficientNetEncoder
    from .pipeline_wuerstchen import WuerstchenGeneratorPipeline, WuerstchenPriorPipeline
