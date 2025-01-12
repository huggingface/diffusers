from ..utils import is_torch_available


if is_torch_available():
    from .layerwise_upcasting import apply_layerwise_upcasting, apply_layerwise_upcasting_hook
