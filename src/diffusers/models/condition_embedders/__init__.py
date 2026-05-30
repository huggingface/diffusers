from ...utils import is_torch_available


if is_torch_available():
    from .condition_embedder_anima import AnimaTextConditioner
